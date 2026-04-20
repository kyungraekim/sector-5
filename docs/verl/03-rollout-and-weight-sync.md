# 03 · Rollout & Weight Sync

This is the core of verl's RL infrastructure: how the **training engine** (Megatron-LM / FSDP) hands updated parameters to the **rollout engine** (SGLang / vLLM) every step, and how the two engines share GPUs without thrashing.

Read this file end-to-end. Skip only if you are never going to touch rollout mode, checkpoint-engine backend, or the `update_weights_bucket_megabytes` knob.

---

## 1. The three deployment modes

Defined in `verl/workers/rollout/replica.py` as `RolloutMode`:

| Mode | Layout | Weight-sync path | When to use |
|------|--------|------------------|-------------|
| `COLOCATED` | Training + rollout on the same GPU set; only one active at a time | In-process handoff after `sleep` → `wake` | Small-scale, research, GPU-poor |
| `HYBRID` | Same node, *separate processes* for trainer and rollout; each owns its own GPUs or alternates via sleep/wake | **CUDA IPC** via `BucketedWeightSender` | Most production setups |
| `STANDALONE` | Trainer and rollout on **separate clusters** | NCCL / NIXL collective from training workers to rollout workers | Very large models, disaggregated serving |

The mode is implicit from `rollout.checkpoint_engine.backend`:
- `backend: naive` → colocated/hybrid with in-process IPC (the default).
- `backend: nccl` / `nixl` / etc. → standalone via a `CheckpointEngine` plugin.

`CheckpointEngineManager` in `verl/checkpoint_engine/base.py:312` draws its topology explicitly in an ASCII diagram you should read:

```
┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
│ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
│ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
│ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
│    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
| ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
│ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
│ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
└────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
     v                                        |    |    |    |    |    |    |    |
     └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
```

`ME` = model engine (training side), `CE` = checkpoint engine. In colocated/hybrid, the CE is collapsed into CUDA IPC; in standalone, the CE is a distinct process communicating over NCCL/NIXL.

---

## 2. The per-step sync — canonical sequence

Driver-side entry: `self.checkpoint_manager.update_weights(global_steps)` called from `ray_trainer.py:1566`.

```python
# verl/checkpoint_engine/base.py:409
@auto_await
async def update_weights(self, global_steps: int = None):
    """Update weights from trainer to rollout replicas."""
    # 0. update weights for sync training with colocated trainer and rollout
    if self.backend == "naive":
        ray.get(self.trainer.update_weights(global_steps=global_steps))
        return
    # ... disaggregated path (NCCL/NIXL) ...
```

The `self.trainer.update_weights(...)` call is an RPC into every `ActorRolloutRefWorker` actor. The worker-side method:

```python
# verl/workers/engine_workers.py:663
@register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
async def update_weights(self, global_steps: int = None):
    # 0. send_weights only for async training with disaggregated trainer and rollout
    if self.config.rollout.checkpoint_engine.backend != "naive":
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)
        return

    set_expandable_segments(False)

    # 1. resume rollout memory (weights were released during sleep)
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["weights"])

    # 2. determine if we need a base weight sync (adapter path only)
    per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon, base_sync_done=True
    )

    do_lora_base_sync = False
    if not self.peft_merge and peft_config is not None:
        self.rollout.sleep_level = 1
        do_lora_base_sync = not self.base_sync_done

    # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
    if do_lora_base_sync:
        per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=False
        )
        await self.rollout.update_weights(per_tensor_param_base, peft_config=peft_config,
                                          base_sync_done=False, global_steps=global_steps)

    await self.rollout.update_weights(per_tensor_param, peft_config=peft_config,
                                      base_sync_done=True, global_steps=global_steps)

    # 3. offload model to cpu
    if self.actor.engine.is_param_offload_enabled:
        self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
    aggressive_empty_cache(force_sync=True)

    # 4. resume kv_cache
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["kv_cache"])

    self.base_sync_done = True
    set_expandable_segments(True)
```

The sequence in plain English:

1. **Disable expandable segments** — avoids PyTorch's caching allocator fragmenting during the weight copy.
2. **Wake the rollout's weight region** (`resume(tags=["weights"])`) — in vLLM/SGLang this is a `wake_up(tags=...)` that re-allocates the weight tensors on the GPU. KV cache stays asleep.
3. **Pull parameters from the trainer** as a generator of `(name, tensor)` pairs (`get_per_tensor_param`). The generator is important — it means tensors are materialized one at a time, so gradient memory + weight memory do not both peak at once.
4. **LoRA branch** — if training with PEFT adapters and they haven't been synced yet as base weights, do a one-time base sync first, then the adapter sync.
5. **Ship the weights** via `rollout.update_weights(generator, ...)`. This is where CUDA IPC happens (see §3).
6. **Offload the trainer to CPU** so the rollout can use the full GPU memory.
7. **Wake the rollout's KV cache** (`resume(tags=["kv_cache"])`) — ready for the next generation step.

---

## 3. Bucketed CUDA IPC — how the tensors actually move

vLLM side: `verl/workers/rollout/vllm_rollout/vllm_rollout.py:162`:

```python
async def update_weights(
    self, weights: Generator[tuple[str, torch.Tensor], None, None],
    global_steps: int = None, **kwargs
):
    """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported)."""
    future = await self._execute_method(
        "update_weights_from_ipc",
        non_block=True,
        kwargs={**kwargs, "use_shm": self.use_shm},
    )

    bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
    sender = BucketedWeightSender(
        zmq_handle=self.zmq_handle,
        bucket_size_mb=bucket_size_mb,
        use_shm=self.use_shm,
    )
    await sender.async_send_weights(weights)

    if future is not None:
        await future

    # reset prefix cache after updating weights
    if self.rollout_rank == 0:
        await self.server_handle.clear_kv_cache.remote()
        if global_steps is not None:
            await self.server_handle.set_global_steps.remote(global_steps)
```

Two actors run concurrently:

- **Sender** (trainer side): `BucketedWeightSender` at `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:73`. Packs named tensors into a fixed-size pinned GPU buffer.
- **Receiver** (rollout worker side): `update_weights_from_ipc` at `verl/workers/rollout/vllm_rollout/utils.py:179` unpacks each bucket into the vLLM model.

### Sender loop

```python
# bucketed_weight_transfer.py:102-154
async def async_send_weights(self, weights):
    from verl.workers.rollout.utils import ensure_async_iterator
    try:
        self._init_socket()
        self._init_buffer()

        offset = 0
        bucket_meta: dict[str, TensorMetadata] = {}
        async for name, weight in ensure_async_iterator(weights):
            # fill the tensor bucket
            if offset + weight.nbytes > self.bucket_size:
                get_torch_device().synchronize()
                self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                self.socket.recv()
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )
            bucket_meta[name] = {"name": name, "shape": weight.shape,
                                  "dtype": weight.dtype, "offset": offset}
            self.buffer[offset : offset + weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += weight.nbytes

        # send the last bucket
        get_torch_device().synchronize()
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
        self.socket.recv()
    finally:
        self._cleanup()
```

Mechanics:

- The communication buffer (`self.buffer`) is a single CUDA-IPC-shared (or SHM on NPU) tensor of size `bucket_size_mb << 20` bytes. Both trainer and rollout open the same handle.
- For each incoming `(name, tensor)`, the sender `memcpy`'s the raw bytes into the buffer at a running offset.
- When the next tensor would overflow the bucket, the sender calls `torch.cuda.synchronize()` (to flush the non-blocking copies), sends a ZMQ REQ with `bucket_meta` describing what's in the buffer, waits for the receiver's REP (ack), then resets `offset=0`.
- At end-of-stream: one final `synchronize` + `{"is_last": True}` message.

Design choices worth calling out:

- **Bucket size is user-tunable.** The relevant config is `rollout.checkpoint_engine.update_weights_bucket_megabytes`. Smaller → less peak GPU memory for the buffer, more ZMQ roundtrips. Larger → fewer roundtrips, more memory pressure during sync. Default is usually 512 MB.
- **No per-weight castings.** Comment at `bucketed_weight_transfer.py:120-124` says: the sender deliberately does *not* downcast to the rollout's dtype, because some params (MoE gates) must stay fp32. The rollout does "cast on demand" per-parameter during `_update_weights`.
- **Fallback to shared memory.** When CUDA IPC is unavailable (Ascend NPU), the buffer is allocated in POSIX shared memory (`/dev/shm`) and the receiver reconstructs tensors via `torch.frombuffer`.

### Receiver loop

`BucketedWeightReceiver` (same file, `:212`) mirrors the sender: `recv_pyobj` to get metadata, slice the shared buffer into individual views, pass the list to `on_bucket_received` (which is `self._update_weights` at `vllm_rollout/utils.py:242`). That method calls into vLLM's `load_weights()` per bucket; once the last bucket is received and `base_sync_done=True`, it optionally runs `process_weights_after_loading` to finalize (e.g. dequantize, re-fuse QKV).

### SGLang side

Analogous, in `verl/workers/rollout/sglang_rollout/sglang_rollout.py:205`:

```python
async def update_weights(self, weights, global_steps=None, **kwargs):
    await self._init_server_adapter()
    async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
        await sgl_update_weights(
            engine=self._engine,
            params_batch=params_batch,
            device_mesh_key="infer_tp",
            device_mesh=self.device_mesh,
        )
```

SGLang buckets by *byte budget* using `get_named_tensor_buckets`, then calls SGLang's native `update_weights_from_tensor` RPC (`sglang_rollout/http_server_engine.py:348` / `:745`). Under the hood SGLang uses its own IPC path.

---

## 4. The sleep/wake primitive

Rollout side (`vllm_rollout.py:147-159`):

```python
async def resume(self, tags: list[str]):
    """Resume rollout weights or kv cache in GPU memory.
    Args:
        tags: weights or kv_cache.
    """
    if self.config.free_cache_engine:
        await self._execute_method("wake_up", kwargs={"tags": tags})

async def release(self):
    """Release weights and kv cache in GPU memory."""
    if self.config.free_cache_engine:
        await self._execute_method("sleep", kwargs={"level": self.sleep_level})
```

Under the hood (server side — `vllm_async_server.py:555-578`):

```python
async def wake_up(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        raise ValueError("wake_up not supported in HYBRID mode")
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.wake_up(tags=self._get_wake_up_tags())
        await self.engine.reset_prefix_cache()

async def sleep(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        await self._sleep_hybrid()
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.sleep(level=1)
```

**Sleep levels**:
- **level 1** — release KV cache, keep weight tensors allocated. Used for LoRA adapter mode (base weights stay in place; only adapters update).
- **level 2** — release *everything*, including weights. Used after a full merge or when the trainer needs the whole GPU.

The `level` is picked in `ActorRolloutRefWorker.update_weights` based on `self.peft_merge` and `peft_config` (`engine_workers.py:697`).

The `tags=["weights"|"kv_cache"]` parameter is finer-grained than sleep level — it lets `resume()` bring back *only* weights (so the trainer can push updates) without re-allocating the KV cache until after the sync completes.

---

## 5. Device mesh and DP/TP layout

In `ActorRolloutRefWorker.__init__` (`engine_workers.py` around `:592`):

```python
infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
infer_pp = rollout_config.pipeline_model_parallel_size
infer_world_size = infer_tp * infer_pp
dp = self.world_size // infer_world_size

rollout_device_mesh = init_device_mesh(
    "cuda", mesh_shape=(dp, infer_tp, infer_pp),
    mesh_dim_names=["dp", "infer_tp", "infer_pp"]
)
```

The trainer has its own mesh (Megatron's `mpu.initialize_model_parallel(...)`). The two meshes are independent but overlapping — both live on the same `torch.distributed` world.

Practically: each rollout replica = one full model copy sharded across `infer_tp × infer_pp` ranks; `dp` replicas run in parallel for throughput. The total GPU count must equal the trainer's TP×PP×DP.

---

## 6. Async rollout + agent loop

`AsyncLLMServerManager` (`verl/experimental/agent_loop/agent_loop.py:107`) is what `generate_sequences` really calls in modern verl — `ServerAdapter.generate_sequences` itself *raises `NotImplementedError`* (`vllm_rollout.py:198-214`) because the SPMD mode was retired in PR #4411.

```python
# agent_loop.py:107
class AsyncLLMServerManager:
    """Manage multiple OpenAI-compatible LLM servers with load balancing."""
    def __init__(self, config, servers, load_balancer_handle): ...

    async def _acquire_server(self, request_id: str):
        server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
        handle = self._server_id_to_handle.get(server_id)
        return server_id, handle
```

Requests are routed by a `GlobalRequestLoadBalancer` (`:65`) that pins a `request_id` to the same server across multiple turns — crucial for prefix-cache reuse in multi-turn agent rollouts. Server-side inflight counts drive the tie-breaker.

### Agent-loop protocol

Subclasses of `AgentLoopBase` (`agent_loop.py:284`) implement:

```python
@abstractmethod
async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    """Run agent loop to interact with LLM server and environment."""
```

Registered loops (in the source):

- `@register("single_turn_agent")` — `single_turn_agent_loop.py:34`. Plain one-shot generation.
- `@register("diffusion_single_turn_agent")` — `single_turn_agent_loop.py:95`. For image/diffusion models.
- `@register("tool_agent")` — `tool_agent_loop.py:88`. Multi-turn with tool calls.

Registry (`agent_loop.py:426`):

```python
_agent_loop_registry: dict[str, dict] = {}
def register(agent_name: str):
    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass
    return decorator
```

So registration is by decorator → lands in a dict → Hydra's `instantiate` later resolves `_target_` to construct the right subclass.

---

## 7. Debugging checklist

Common failure modes and where to look:

| Symptom | Likely cause | Check |
|---------|--------------|-------|
| OOM during `update_weights` | `update_weights_bucket_megabytes` too high | Lower it; or use `sleep_level=2` |
| `Weight X is too large to fit in the bucket` | Giant embedding tensor | Increase bucket size, or wait for the TODO in `bucketed_weight_transfer.py:135` (slice embedding chunks) |
| Hanging after generation | ZMQ socket not cleaned up | Check `/tmp/rl-colocate-zmq-*.sock`; restart |
| Weight sync succeeds but rollout produces garbage | dtype mismatch; receiver didn't cast | Verify `param_dtype` config matches rollout dtype |
| "NotImplementedError: ServerAdapter does not support synchronous generate_sequences" | Using old sync path | Switch to async via `AsyncLLMServerManager` / `AgentLoopBase` |
| Multi-turn prompts cache-miss on every turn | Sticky session not configured | Pass the same `request_id` across turns |

Enable verbose tracing: `docs/advance/rollout_trace.rst` — spans for every sleep/resume/update, logged via `@rollout_trace_op`. Great for pinpointing which step in §2 is slow.

---

## 8. Summary

- One training step ≈ one full (sleep → weight copy → wake) cycle per rollout replica.
- The ordering `release → get_per_tensor_param → update_weights → resume(kv_cache)` is invariant; violating it causes silent KV-cache corruption or wasted memory.
- The bucket size is the single most impactful knob for weight-sync latency. Tune per cluster.
- On modern verl (post-PR #4411), **all** generation goes through `AsyncLLMServerManager` + an `AgentLoopBase` subclass; the old sync `generate_sequences` path is a hard error.
