# Knowledge Distillation: End-to-End Execution Flow

> **Scope**: Traces the complete execution path from CLI invocation to optimizer step
> **Audience**: Engineers debugging, profiling, or extending the distillation pipeline

---

## Phase 1: CLI & Argument Parsing

**What happens**: `pretrain_gpt.py` registers ModelOpt-specific CLI arguments and calls into
the Megatron training entry point.

**Entry point**: `pretrain_gpt.py:269-277`

```python
pretrain(
    train_valid_test_datasets_provider,
    partial(model_provider, gpt_builder),
    ModelType.encoder_or_decoder,
    forward_step,
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
    store=store,
)
```

The `extra_args_provider=add_modelopt_args` callback
(`megatron/post_training/arguments.py:4-122`) adds distillation-specific flags to the argument
parser:

| Flag | Purpose |
|------|---------|
| `--export-kd-teacher-load` | Path to teacher checkpoint directory |
| `--export-kd-cfg` | Path to distillation YAML config file |
| `--teacher-model-config` | Path to teacher model config (default: `${teacher_load}/model_config.yaml`) |
| `--export-kd-teacher-ckpt-format` | Teacher checkpoint format if different from student's |

`pretrain()` (`megatron/training/training.py:579`) calls `initialize_megatron()` which parses
all arguments (including the extra ModelOpt ones) and sets up distributed process groups.

---

## Phase 2: Model Builder Swap

**What happens**: The `get_model()` function detects that distillation is enabled and swaps the
default model builder for ModelOpt's builder.

**Location**: `megatron/training/training.py:945-961`

```python
def get_model(model_provider_func, ...):
    args = get_args()

    if has_nvidia_modelopt:
        from megatron.post_training.checkpointing import has_modelopt_state
        if args.load is not None and has_modelopt_state(args.load):
            args.modelopt_enabled = True
        elif getattr(args, "export_kd_teacher_load", None):
            args.modelopt_enabled = True
```

The `model_provider_func` is the `model_provider()` function from `model_provider.py:24-67`.
When `modelopt_enabled` is set, it swaps the builder:

```python
# model_provider.py:63-65
if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):
    model_builder = modelopt_gpt_mamba_builder
```

This replaces the default `gpt_builder` with `modelopt_gpt_mamba_builder`
(`megatron/post_training/model_builder.py:154-341`), which handles the entire distillation
setup pipeline described in the following phases.

---

## Phase 3: Teacher Config Loading

**What happens**: The teacher model's architecture configuration is read from a NeMo-format
YAML file and translated to Megatron argument names.

**Location**: `megatron/post_training/model_builder.py:47-104`

```python
def _load_teacher_model_config(checkpoint_path: str) -> Namespace:
    config_path = os.path.join(checkpoint_path, "model_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
```

**NeMo-to-Megatron field translations** (lines 76-98):

| NeMo Field | Megatron Field |
|-----------|---------------|
| `encoder_seq_length` | `seq_length` |
| `bias` | `disable_bias_linear` (inverted) |
| `activation: "swiglu"` | `swiglu: True` |
| `position_embedding_type: null` | `use_rotary_position_embeddings: True` |
| `share_embeddings_and_output_weights` | `untie_embeddings_and_output_weights` (inverted) |
| `masked_softmax_fusion` | `no_masked_softmax_fusion` (inverted) |
| `normalization: "layernorm1p"` | `apply_layernorm_1p: True` |

The function starts from a copy of the student's args and overlays the teacher-specific
fields, producing a `Namespace` that can be fed to `core_transformer_config_from_args()`.

---

## Phase 4: Student Model Creation

**What happens**: The student model is constructed as a standard `MCoreGPTModel` (or
`MCoreMambaModel`), and any existing ModelOpt state is restored.

**Location**: `megatron/post_training/model_builder.py:242-301`

```python
# Line 257
model = MCoreGPTModel(config=config, **model_kwargs)

# Line 298-299: Restore ModelOpt state (e.g., quantization scales, PEFT adapters)
if args.load is not None:
    load_modelopt_state(model=model)
```

**Why ModelOpt state is loaded here**: ModelOpt can create additional trainable parameters
(e.g., quantization scales). These must be registered before the optimizer is created, which
happens right after `model_provider` returns. Loading ModelOpt state during normal checkpoint
loading would be too late.

---

## Phase 5: Teacher Model Loading

**What happens**: A teacher model instance is created with the teacher's architecture config,
and its weights are loaded from a checkpoint.

**Location**: `megatron/post_training/model_builder.py:107-151`

```python
def _load_teacher_model(config, config_raw, model_kwargs):
    teacher = MCoreGPTModel(config=config, **model_kwargs)

    # WAR: Temporarily set args.finetune=True to bypass checkpoint validation
    original_args_finetune, original_ckpt_format = args.finetune, args.ckpt_format
    args.finetune = True
    if args.export_kd_teacher_ckpt_format is not None:
        args.ckpt_format = args.export_kd_teacher_ckpt_format
    load_modelopt_checkpoint([teacher], load_arg='export_kd_teacher_load')
    args.finetune, args.ckpt_format = original_args_finetune, original_ckpt_format
```

**The `finetune=True` workaround** (lines 143-148): Megatron's checkpoint loader validates
saved args and RNG state against current args. Teacher and student have different architectures,
so this validation would fail. Setting `finetune=True` temporarily bypasses these checks.

The teacher's layer spec is recreated since it depends on the teacher's `num_layers`, not the
student's (line 129-135).

---

## Phase 6: Distillation Wrapping

**What happens**: The student model is wrapped in a `DistillationModel` that pairs it with the
teacher, and Megatron-specific patches are applied.

**Location**: `megatron/post_training/model_builder.py:324-338`

### Step 6a: Build criterion and balancer

```python
distill_cfg = mtd_mcore.setup_distillation_config(
    args.export_kd_cfg, student_cfg=config, teacher_cfg=teacher_config
)
```

`setup_distillation_config()` (`plugins/megatron.py:98-161`) reads the YAML config and
constructs:
- **Criterion dict**: Maps `(student_layer_name, teacher_layer_name)` tuples to loss function
  instances (`LogitsKLLoss`, `TopKLogitsKLLoss`, `HiddenStateCosineLoss`, or `MSELoss`)
- **Loss balancer**: `LogitsAndIntermediatesLossBalancer` with the configured
  `kd_loss_scale` and `skip_lm_loss` settings

For pipeline parallelism, layer indices in intermediate pair names are adjusted by
`_adjust_layer_index_for_pp()` (`plugins/megatron.py:164-180`) to account for layer offset on
the current PP rank.

### Step 6b: Convert model

```python
model = mtd.convert(model, mode=[("kd_loss", kd_config)])
```

This calls `_convert_for_kd()` (`modelopt/torch/distill/mode.py:144-196`) which:

1. Initializes the teacher from the config (already instantiated in this case)
2. Registers the student's class in `DistillationDMRegistry`
3. Converts the student to a `DistillationModel` via `model_registry.convert(student)`
4. Calls `DistillationModel.modify()` which:
   - Assigns `_teacher_model` and freezes it via `requires_grad_(False)`
     (`distillation_model.py:108`)
   - Resolves layer name strings to actual module references in `_layers_to_loss`
   - Registers forward hooks on all student/teacher layer pairs
     (`distillation_model.py:113-120`)

### Step 6c: Apply Megatron-Core patches

```python
mtd_mcore.adjust_distillation_model_for_mcore(model, distill_cfg)
```

`adjust_distillation_model_for_mcore()` (`plugins/megatron.py:558-616`) monkey-patches several
methods:

| Patch | Purpose |
|-------|---------|
| `sharded_state_dict` | Hides teacher during distributed checkpoint save |
| `compute_language_model_loss` (student) | Returns zeros when `skip_lm_loss=True` during training |
| `compute_language_model_loss` (teacher) | Always returns zeros (teacher LM loss never needed) |
| `forward` | Concatenates student+teacher outputs for PP inter-rank communication |
| `set_input_tensor` | Splits received tensor into student/teacher portions |

---

## Phase 7: Forward Pass

**What happens**: During each training micro-batch, the teacher runs first (no gradients),
then the student runs. Forward hooks capture intermediate activations.

### Standard case (PP=1)

**Location**: `modelopt/torch/distill/distillation_model.py:215-241`

```python
def forward(self, *args, **kwargs):
    if not self._only_student_fwd:
        # Teacher forward: no activations saved for autograd
        with torch.no_grad():
            self._teacher_model.eval()
            teacher_output = self._teacher_model(*args, **kwargs)

    # Student forward: normal autograd tracking
    student_output = super().forward(*args, **kwargs)
    return student_output
```

**Hook execution**: During both forwards, registered forward hooks fire on designated layers:

- `student_output_capture_fwd_hook` (`distillation_model.py:295-307`): Saves
  `module._intermediate_output = output`
- `teacher_output_capture_fwd_hook` (`distillation_model.py:310-321`): Same, with a warning
  if output already stored (expected in eval mode)

### Pipeline-parallel case (PP>1)

**Location**: `plugins/megatron.py:604-616` (patched `_forward`)

```python
def _forward(self, *args, **kwargs):
    with torch.no_grad():
        self._teacher_model.eval()
        teacher_output = self._teacher_model(*args, **kwargs)
    with self.only_student_forward():
        student_output = type(self).forward(self, *args, **kwargs)

    if not parallel_state.is_pipeline_last_stage():
        # Non-last stages: concatenate for single P2P send
        return torch.cat([student_output, teacher_output], dim=-1)
    else:
        return student_output
```

On receiving stages, `set_input_tensor` splits the concatenated tensor:

```python
def _set_input_tensor(self, input_tensors):
    teacher_inputs = [t[..., self._tensor_split_idx:] for t in input_tensors]
    student_inputs = [t[..., :self._tensor_split_idx] for t in input_tensors]
    type(self).set_input_tensor(self.teacher_model, teacher_inputs)
    type(self).set_input_tensor(self, student_inputs)
```

---

## Phase 8: Loss Computation

**What happens**: The LM loss is computed (or skipped), then KD losses are computed from
captured activations, and the loss balancer produces the final scalar loss.

**Location**: `megatron/post_training/loss_func.py:39-72`

```python
def loss_func(loss_mask, output_tensor, model):
    model = unwrap_model(model)

    # Standard LM loss (may be zeros if skip_lm_loss=True)
    loss_lm = _mask_loss(output_tensor, loss_mask)
    loss = loss_lm

    if args.export_kd_teacher_load:
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
        )
        if model.training:
            loss = losses["kd_loss"]
```

### Inside `compute_kd_loss()`

**Location**: `modelopt/torch/distill/distillation_model.py:243-292`

1. **Collect individual losses**: Iterates `_layers_to_loss`, reads `_intermediate_output`
   from each student/teacher layer pair, calls the loss function, applies
   `loss_reduction_fn` (which runs `_mask_loss` for proper loss masking):

   ```python
   for i, ((student_layer, teacher_layer), loss_fn) in enumerate(self._layers_to_loss.items()):
       out_s = student_layer._intermediate_output
       out_t = teacher_layer._intermediate_output
       loss = loss_fn(out_s, out_t)
       if loss_reduction_fn is not None:
           loss = loss_reduction_fn(loss)
       loss_dict[f"{loss_fn.__class__.__name__}_{i}"] = loss
   ```

2. **Balance losses**: Delegates to `LogitsAndIntermediatesLossBalancer.forward()`
   (`plugins/megatron.py:480-516`):

   ```python
   # Separate logits loss from intermediate losses
   logits_loss = loss_dict.pop(logits_key)
   intermediate_loss = sum(loss_dict.values()) / max(len(loss_dict), 1)

   # Dynamic scale: match intermediate magnitude to logits
   dynamic_scale = logits_loss.detach() / intermediate_loss.detach()
   intermediate_loss_scaled = intermediate_loss * dynamic_scale

   if skip_original_loss:
       total_loss = logits_loss + intermediate_loss_scaled
   else:
       kd_loss = logits_loss + intermediate_loss_scaled
       # Dynamic scale: match KD magnitude to original LM loss
       kd_loss *= original_loss.detach() / kd_loss.detach()
       total_loss = original_loss + kd_loss * kd_loss_scale
   ```

### Loss masking (`_mask_loss`)

**Location**: `megatron/post_training/loss_func.py:13-36`

Handles three cases based on flags returned by the loss function's `post_forward()`:

- **Standard**: `losses * loss_mask` then sum
- **TP-reduce** (`tp_reduce=True`): Extra `all_reduce` across TP group after masking
- **Sequence-parallel** (`is_sequence_parallel=True`): Splits loss mask via
  `tensor_split` across TP ranks before masking, then `all_reduce`

---

## Phase 9: Backward & Optimizer

**What happens**: Gradients flow backward through the KD loss, but only student parameters
receive updates.

### Gradient isolation

Two mechanisms ensure the teacher receives no gradient updates:

1. **`requires_grad_(False)`** on all teacher parameters
   (`distillation_model.py:108`) — prevents gradient accumulation in teacher parameter
   tensors.

2. **`torch.no_grad()` context** during teacher forward (`distillation_model.py:230`) —
   prevents PyTorch from building the autograd graph for the teacher's forward pass entirely,
   meaning no activations are saved for backward.

The teacher's output tensors are `.detach()`ed in `BaseLoss.pre_forward()`
(`plugins/megatron.py:210`), severing any remaining gradient connection.

### Optimizer step

The optimizer (created by Megatron's `setup_model_and_optimizer()`) only holds references to
`model.parameters()` that have `requires_grad=True` — i.e., only student parameters.
Teacher parameters are excluded from the optimizer entirely.

The standard Megatron training loop then:
1. Calls `loss.backward()` on the scalar `kd_loss`
2. Reduces gradients across DP ranks
3. Clips gradients
4. Steps the optimizer (updates student only)
5. Steps the learning rate scheduler

---

## Summary: Call Graph

```
pretrain_gpt.py::forward_step()
  └─ model(tokens, position_ids, attention_mask, labels)
       └─ DistillationModel.forward()  [patched for PP]
            ├─ torch.no_grad():
            │    └─ teacher.forward(tokens, ...)    → hooks capture teacher activations
            └─ student.forward(tokens, ...)          → hooks capture student activations
                 └─ GPTModel.forward()
                      └─ compute_language_model_loss()  [patched: may return zeros]
                           └─ output_tensor (LM loss per token)

pretrain_gpt.py::loss_func()  [called on pipeline last stage]
  └─ loss_func_modelopt(loss_mask, output_tensor, model)
       ├─ _mask_loss(output_tensor, loss_mask)  → loss_lm (scalar)
       └─ model.compute_kd_loss(student_loss=loss_lm, loss_reduction_fn=_mask_loss)
            ├─ For each (student_layer, teacher_layer) pair:
            │    └─ loss_fn(student._intermediate_output, teacher._intermediate_output)
            │         └─ _mask_loss(loss_per_token, loss_mask)  → scalar
            └─ LogitsAndIntermediatesLossBalancer(loss_dict)
                 └─ returns {"kd_loss": total, "logits_loss": ..., "intermediate_loss": ...}

training.py::train_step()
  └─ loss.backward()      # gradients flow to student only
  └─ optimizer.step()     # updates student parameters
```
