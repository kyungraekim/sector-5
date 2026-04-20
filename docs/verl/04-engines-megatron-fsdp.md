# 04 · Training Engines — Megatron (primary) & FSDP (secondary)

verl unifies training backends behind a single `BaseEngine` contract. Picking a backend is a single config line:

```
model_engine: megatron   # or: fsdp | torchtitan | veomni | mindspeed
```

This file shows the `BaseEngine` interface, then walks through `MegatronEngine` in detail (the production backend for large models), then summarizes how FSDP fits into the same contract.

---

## 1. The `BaseEngine` contract

File: `verl/workers/engine/base.py:29`.

```python
class BaseEngine:
    """Abstract base class defining the interface for model training engines."""

    def initialize(self):
        """Instantiate or load the model, optimizer, and learning rate scheduler."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool: ...

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool: ...

    def train_mode(self, **kwargs):
        """Context manager entry for switching the engine and model into training mode."""
        raise NotImplementedError

    def eval_mode(self, **kwargs):
        """Context manager entry for switching the engine and model into evaluation mode."""
        raise NotImplementedError

    def optimizer_zero_grad(self): ...
    def optimizer_step(self): ...
    def lr_scheduler_step(self): ...

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable,
                               forward_only=False) -> Any:
        """Perform a forward pass and optionally a backward pass on a batch of data."""
        raise NotImplementedError

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        """Perform a training step on a batch of data."""
        raise NotImplementedError
```

There is also a hidden but critical method every backend must implement: `get_per_tensor_param(layered_summon=..., base_sync_done=...)` returning a `(generator_of_(name, tensor), peft_config)` tuple. This is what the rollout weight-sync path consumes (see `03-rollout-and-weight-sync.md`). It's what lets the engine swap implementations without the rollout caring.

### The engine selector

`EngineRegistry` in the same file lets `ActorRolloutRefWorker` do:

```python
engine_cls = EngineRegistry.get(config.model_engine)   # "megatron" → MegatronEngine
self.engine = engine_cls(model_config, engine_config, optimizer_config, checkpoint_config)
self.engine.initialize()
```

No conditional imports in the trainer — everything is dispatched by name.

---

## 2. `MegatronEngine`

File: `verl/workers/engine/megatron/transformer_impl.py:71`.

### Construction

```python
# :71-125
class MegatronEngine(BaseEngine):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        self._init_device_mesh()
        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        # QAT configuration
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)

        # Router replay configuration for MoE models
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        ...
```

### Parallelism setup

`_init_device_mesh` calls Megatron's `mpu.initialize_model_parallel` (`:127-154`) with:

```python
mpu.initialize_model_parallel(
    tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
    pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
    use_sharp=False,
    context_parallel_size=self.engine_config.context_parallel_size,
    expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
    expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
    nccl_communicator_config_path=None,
    **extra_args,
)
```

What each dimension means:

| Axis | Config key | Purpose |
|------|-----------|---------|
| TP | `tensor_model_parallel_size` | Shards every matmul; limits TP by intra-node NVLink bandwidth |
| PP | `pipeline_model_parallel_size` | Splits layers across GPUs; uses 1F1B or interleaved schedule |
| VP | `virtual_pipeline_model_parallel_size` | Interleaved PP — more micro-batch parallelism, smaller bubbles |
| CP | `context_parallel_size` | Shards sequence dim across GPUs for long contexts |
| EP | `expert_model_parallel_size` | MoE expert sharding |
| ETP | `expert_tensor_parallel_size` | Tensor-parallel *within* each MoE expert |
| DP | (implicit) | `world_size / (TP·PP·CP·EP)` |

World size must equal the product. If `dynamic_context_parallel=True`, verl also validates `max_seqlen_per_dp_cp_rank` (`:134-142`) for variable-length CP scheduling.

### Model construction via `AutoBridge`

Two paths in `_build_tf_config` (`:156-257`):

**Vanilla bridge** (`:170-185`):

```python
if self.vanilla_bridge:
    from verl.models.mcore.mbridge import AutoBridge
    bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
    if self.engine_config.dynamic_context_parallel:
        override_transformer_config["max_seqlen_per_dp_cp_rank"] = self.engine_config.max_seqlen_per_dp_cp_rank
        override_transformer_config["dynamic_context_parallel"] = False
        override_transformer_config["context_parallel_size"] = mpu.get_data_parallel_world_size()
    bridge.set_extra_args(**override_transformer_config)
    tf_config = bridge.config
```

**Non-vanilla (Megatron-Bridge)** (`:186-235`):

```python
else:
    from verl.models.mcore.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(
        self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
    )
    provider = bridge.to_megatron_provider(load_weights=False)

    provider.params_dtype = self.param_dtype
    provider.fp16 = self.param_dtype == torch.float16
    provider.bf16 = self.param_dtype == torch.bfloat16
    provider.tensor_model_parallel_size = self.engine_config.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = self.engine_config.pipeline_model_parallel_size
    provider.expert_model_parallel_size = self.engine_config.expert_model_parallel_size
    provider.expert_tensor_parallel_size = self.engine_config.expert_tensor_parallel_size
    ...
    provider.attention_backend = AttnBackend.flash
    provider.variable_seq_lengths = True
    provider.moe_token_dispatcher_type = "alltoall"

    if self._qat_enabled:
        from verl.utils.modelopt import patch_provider_for_qat
        patch_provider_for_qat(provider)

    provider.finalize()
    self.provider = provider
```

`AutoBridge` is the translation layer: an HF config → a Megatron `TransformerConfig` (the vanilla path) or a Megatron-Bridge `provider` (the non-vanilla path, which supports QAT and finer control).

The layer-name mapping `{"qkv_layer_name": "self_attention.linear_qkv.", "gate_proj_layer_name": "linear_fc1."}` (`:96-99`) is how verl maps Megatron's parameter names back to HF-style names when streaming weights to the rollout — essential for `get_per_tensor_param`.

### Forward / backward

`MegatronEngine.train_batch` wraps Megatron's pipeline schedule:

```python
# imported at transformer_impl.py:24
from megatron.core.pipeline_parallel import get_forward_backward_func
```

`get_forward_backward_func()` returns one of Megatron's pipeline schedules (1F1B, interleaved 1F1B, `forward_backward_pipelining_with_interleaving`, or non-pipelined). The schedule takes a list of micro-batches and runs forward+backward passes in the right order to fill the pipeline.

Key micro-batching logic lives in `verl/workers/engine/utils.py::prepare_micro_batches` — splits the mini-batch into micro-batches respecting dynamic-batch size budget (`ppo_max_token_len_per_gpu`).

The loss function is injected from the caller:

```python
# engine_workers.py:337 — TrainingWorker.train_batch
with self.engine.train_mode(disable_auto_offload=disable_auto_offload):
    output = self.engine.train_batch(data, loss_function=self.loss_fn)
```

`self.loss_fn` is `ppo_loss` / `sft_loss` / `value_loss` / etc. from `verl/workers/utils/losses.py` (see `05-losses-and-updates.md`).

### MoE / Router-replay

verl supports router-replay for MoE: the router outputs during *rollout* are captured and replayed during the trainer's old-logprob computation, so the same experts are activated for the same tokens. Controlled by `engine_config.router_replay.mode` (options like `R2`, `R3`, `disabled`).

If `router_replay.mode != "disabled"`, `apply_router_replay_patch()` (`:116-118`) monkey-patches Megatron's router to record and replay. The `@_with_routing_replay_flag(enabled=True)` decorators in `engine_workers.py` (on `update_actor`, `compute_log_prob`) are what flip replay on for trainer-side passes.

### QAT (Quantization-Aware Training)

If `qat.enable=True`:
- Must use non-vanilla bridge (`:106-111`).
- `patch_provider_for_qat(provider)` (`:222-224`) swaps in quantized layer specs.
- Post-weight-load hook (`process_weights_after_loading` on the rollout side) handles dequantization during inference.

### Offloading

`self._is_offload_param`, `_is_offload_grad`, `_is_offload_optimizer` control per-tensor offload-to-CPU. The `engine.to("cpu", model=True, optimizer=False, grad=False)` call in `update_weights` (`engine_workers.py:717`) uses these — only parameters are offloaded between steps; optimizer state and gradients stay on GPU (Megatron manages them via its distributed optimizer).

---

## 3. FSDP (secondary)

Folder: `verl/workers/engine/fsdp/`. Most shipped recipes in `examples/` default to FSDP because it's trivially reproducible — no Megatron build dependency.

What FSDP provides:

- `FSDPEngine` implements the same `BaseEngine` interface.
- Uses PyTorch FSDP for `FULL_SHARD` or `HYBRID_SHARD` — each parameter is sharded across DP ranks, gathered on demand during forward.
- `get_per_tensor_param` uses `FSDP.summon_full_params(module, writeback=False)` (or a layered variant for very large models — `layered_summon=True` gathers one transformer layer at a time to keep peak memory low).
- Sharding-manager siblings exist per rollout backend: `FSDPSGLangShardingManager`, `FSDPvLLMShardingManager` — these are the glue that packages gathered tensors into `(name, tensor)` pairs for the rollout's weight-loader.

Notable FSDP-specific knobs (`actor.fsdp_config`):

| Knob | Effect |
|------|--------|
| `param_offload`, `optimizer_offload` | Offload sharded params/optimizer state to CPU between steps |
| `fsdp_size` | How many ranks form the sharding group (the rest become replica/DP) — enables HYBRID_SHARD |
| `use_orig_params` | Required for mixed-precision + activation checkpointing in some cases |
| `forward_prefetch`, `backward_prefetch` | Comm-compute overlap knobs |

### FSDP vs Megatron — when to pick which

| Criterion | FSDP | Megatron |
|-----------|------|----------|
| Model size that fits comfortably | ≤ 70B dense, ≤ 200B MoE | 70B+ dense, 400B+ MoE |
| Setup complexity | Pip install and go | Needs Megatron-LM / Megatron-Core built |
| Context parallelism (long seq) | Limited | First-class (CP dim) |
| MoE expert parallelism | Via PyTorch MoE libs | First-class (EP + ETP) |
| Pipeline parallelism | No | Yes (PP + VP) |
| Typical use in verl recipes | GSM8K, math_dapo, geo3k, most 7B-32B runs | DeepSeek 671B, Moonlight 16B MoE, Qwen 32B MoE |

Examples: compare `examples/ppo_trainer/run_qwen2-7b_sglang_seq_balance.sh` (FSDP) against `examples/grpo_trainer/run_deepseek671b_math_megatron_80gb.sh` (Megatron).

---

## 4. Other backends (brief)

| Backend | Lives in | When to use |
|---------|----------|-------------|
| `torchtitan` | `verl/workers/engine/torchtitan/` | PyTorch-native replacement for Megatron; simpler build |
| `veomni` | `verl/workers/engine/veomni/` | Bytedance's in-house multimodal-friendly engine |
| `mindspeed` | `verl/workers/engine/mindspeed/` | Ascend NPU-optimized |

All four expose the same `BaseEngine` surface. The trainer and rollout are oblivious.

---

## 5. The weight-sync contract — what every backend must implement

Summarizing what it takes to add a new training engine:

```python
class MyEngine(BaseEngine):
    def initialize(self): ...
    def train_mode(self, **kw): ...  # context manager
    def eval_mode(self, **kw): ...   # context manager
    def forward_backward_batch(self, data, loss_function, forward_only=False): ...
    def train_batch(self, data, loss_function): ...
    def optimizer_zero_grad(self): ...
    def optimizer_step(self): ...
    def lr_scheduler_step(self): ...
    def to(self, device, *, model=True, optimizer=True, grad=True): ...

    # Critical for weight sync:
    def get_per_tensor_param(self, layered_summon=False, base_sync_done=True) \
        -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        """Yield (hf_param_name, full_tensor_on_gpu). peft_config describes adapters if any."""
        ...

    @property
    def is_param_offload_enabled(self) -> bool: ...
    @property
    def is_optimizer_offload_enabled(self) -> bool: ...
```

Register with `EngineRegistry` (decorator or explicit `EngineRegistry.register(...)` — look at how `MegatronEngine` and `FSDPEngine` do it). Expose the matching `*ShardingManager` variant for each rollout backend you want to support.

### Table — backend × rollout combinations that ship

| Training \\ Rollout | vLLM | SGLang | TRT-LLM | naive / HF |
|----|----|----|----|----|
| FSDP | ✅ | ✅ | ✅ | ✅ |
| Megatron | ✅ | ✅ | ⚠️ limited | ✅ |
| torchtitan | ✅ | ✅ | — | ✅ |
| veomni | ✅ | ⚠️ | — | ✅ |
| mindspeed (NPU) | — | ✅ | — | ✅ |

(`⚠️` = exists but less battle-tested; `—` = not implemented.)

---

## 6. Where to read next

- `05-losses-and-updates.md` — what happens inside `train_batch(loss_function=ppo_loss)`.
- `03-rollout-and-weight-sync.md` — the other half of `get_per_tensor_param`.
- `docs/advance/megatron_extension.rst` (upstream) — for adding new Megatron model classes.
- `docs/advance/fsdp_extension.rst` (upstream) — FSDP equivalents.
