# Knowledge Distillation: GPU Optimizations Deep Dive

> **Scope**: Every GPU optimization in the Megatron-LM + ModelOpt distillation pipeline
> **Audience**: Engineers optimizing distillation training performance or extending the system

---

## 1. Teacher Gradient Elimination (Two-Layer Defense)

### What

Prevents GPU memory from being spent on teacher model gradient computation.

### Why

Without this, PyTorch would save all intermediate activations during the teacher's forward
pass to support backpropagation — effectively doubling the activation memory.

### How

Two complementary mechanisms:

**Layer 1: `requires_grad_(False)`** — `distillation_model.py:108`

```python
self._teacher_model.requires_grad_(False)
```

This prevents gradient *accumulation* in teacher parameter `.grad` tensors. However, PyTorch
will still save intermediate activations in the autograd graph if the forward pass is inside a
gradient-enabled context.

**Layer 2: `torch.no_grad()` context** — `distillation_model.py:225-233`

```python
# no_grad() context lets pytorch know not to save activations for
# teacher models in memory as there won't be any gradient updates applied
# to these layers. This consumes less memory than just freezing teacher model weights.
with torch.no_grad():
    self._teacher_model.eval()
    teacher_output = self._teacher_model(*args, **kwargs)
```

The comment at line 227-229 explains the crucial distinction: `no_grad()` prevents the
*autograd graph from being built at all*, so no activations are saved. Just freezing weights
(layer 1 alone) would still cause PyTorch to save activations since the forward context is
gradient-enabled.

**Layer 3: `targets.detach()`** — `plugins/megatron.py:210`

```python
def pre_forward(self, predictions, targets):
    targets = targets.detach()
    return predictions, targets
```

The teacher's captured output is detached before entering the loss function, severing any
residual gradient connection.

### Impact

Approximately 50% memory savings from not storing teacher activation tensors. The teacher
forward pass becomes essentially "free" from a memory perspective — only the final output
tensor (used as the loss target) persists.

---

## 2. Teacher LM Loss Skip

### What

Bypasses the expensive cross-entropy loss computation for the teacher model, and optionally
for the student model.

### Why

The teacher's language-model loss is never used — the teacher is frozen and we only need its
intermediate activations and output logits, not its loss value. Computing cross-entropy over a
large vocabulary (e.g., 128K tokens) is expensive and wasteful.

### How

`adjust_distillation_model_for_mcore()` patches `compute_language_model_loss` on both models:

**Teacher** — always returns zeros (`plugins/megatron.py:579-584`):

```python
def _compute_teacher_lm_loss(self, labels, logits) -> Tensor:
    return torch.zeros_like(labels, dtype=logits.dtype)

model.teacher_model.compute_language_model_loss = MethodType(
    _compute_teacher_lm_loss, model.teacher_model
)
```

**Student** — returns zeros when `skip_lm_loss=True` during training
(`plugins/megatron.py:571-576`):

```python
def _compute_student_lm_loss(self, labels, logits) -> Tensor:
    if distill_cfg.skip_lm_loss and self.training:
        return torch.zeros_like(labels, dtype=logits.dtype)
    return type(self).compute_language_model_loss(self, labels, logits)
```

### Impact

Avoids a full vocabulary-dimension softmax + cross-entropy computation per micro-batch for the
teacher (always) and the student (when `skip_lm_loss=True`). For a 128K vocabulary, this saves
a non-trivial amount of compute and the associated temporary memory allocations.

---

## 3. Tensor-Parallel Aware KL-Divergence

### What

Computes KL-divergence loss correctly when the vocabulary dimension is sharded across
tensor-parallel (TP) ranks, without gathering the full vocabulary tensor.

### Why

With TP, each rank holds only `V/TP` elements of the vocabulary dimension. A naive softmax on
local logits would produce incorrect probabilities since the normalization denominator must
account for all `V` elements. Gathering the full vocabulary would be memory-prohibitive for
large vocabularies.

### How

`LogitsKLLoss.forward()` — `plugins/megatron.py:309-370`

The implementation computes a **distributed softmax** in four steps:

1. **Global max for stability**: Each rank computes its local max, then `all_reduce(MAX)`
   finds the global max. Both student and teacher logits are shifted by this global max.

   ```python
   teacher_logits_max, _ = torch.max(output_teacher, dim=-1, keepdim=True)
   torch.distributed.all_reduce(teacher_logits_max, op=ReduceOp.MAX, group=tp_group)
   output_teacher -= teacher_logits_max
   ```

2. **Distributed denominator**: Each rank computes local `exp` sums, then uses
   `dist_nn.functional.all_reduce` (not `torch.distributed.all_reduce`) to sum denominators
   across ranks:

   ```python
   denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1, keepdim=True)
   denom_teacher = dist_nn.functional.all_reduce(denom_teacher, group=tp_group)
   ```

   **Critical detail**: `dist_nn.functional.all_reduce` is a **differentiable** operation
   that preserves the gradient graph, unlike `torch.distributed.all_reduce` which operates
   in-place and breaks gradients. This is necessary because the student's log-probabilities
   must be differentiable for backpropagation.

3. **Local log-softmax**: Each rank computes `logits - log(global_denominator)`:

   ```python
   teacher_log_prob = output_teacher - torch.log(denom_teacher)
   student_log_prob = output_student - torch.log(denom_student)
   ```

4. **Local KL-divergence**: Each rank computes `kl_div(p, q)` on its local shard, then
   `post_forward()` flags `tp_reduce=True` so that `_mask_loss()` performs a final
   all-reduce to sum the KL contributions across ranks.

### Impact

Each rank operates on `V/TP` vocabulary elements instead of `V`. For a 128K vocabulary with
TP=8, each rank handles only 16K elements — a significant reduction in memory and compute for
the softmax and KL-divergence operations.

---

## 4. Top-K KL-Divergence Loss

### What

Restricts the KL-divergence computation to only the teacher's top-K vocabulary entries,
avoiding computation over the entire vocabulary.

### Why

For large vocabularies, most probability mass is concentrated in a small number of tokens. The
long tail contributes negligible gradient signal but dominates the computational cost.

### How

`TopKLogitsKLLoss.forward()` — `plugins/megatron.py:398-458`

1. **Local top-K extraction**: Each TP rank takes its top-K entries from teacher logits and
   gathers the corresponding student logits:

   ```python
   local_top_k = min(self.top_k, targets.size(-1))
   top_teacher_vals, top_idx = torch.topk(output_teacher, local_top_k, dim=-1)
   top_student_vals = torch.gather(output_student, dim=-1, index=top_idx)
   ```

2. **Differentiable all-gather**: Uses `dist_nn.functional.all_gather` (gradient-preserving)
   to collect all local top-K candidates:

   ```python
   all_teacher_vals = dist_nn.functional.all_gather(top_teacher_vals.contiguous(), group=tp_group)
   all_student_vals = dist_nn.functional.all_gather(top_student_vals.contiguous(), group=tp_group)
   ```

3. **Global top-K selection**: From `local_K * TP` candidates, select the true top-K based
   on teacher values:

   ```python
   global_top_vals, global_top_idx = torch.topk(all_teacher_vals, self.top_k, dim=-1)
   final_student_logits = torch.gather(all_student_vals, dim=-1, index=global_top_idx)
   ```

4. **Dense softmax on reduced set**: Since all ranks now have identical top-K entries, a
   standard `F.log_softmax` + `F.kl_div` is computed locally — no additional all-reduce
   needed (`tp_reduce=False`):

   ```python
   p = F.log_softmax(final_student_logits, dim=-1)
   q = F.log_softmax(final_teacher_logits, dim=-1)
   loss = torch.sum(F.kl_div(p, q, reduction="none", log_target=True), dim=-1)
   return self.post_forward(loss, tp_reduce=False)
   ```

### Impact

Reduces the vocabulary dimension for softmax and KL computation from `V` (e.g., 128K) to `K`
(e.g., 1024). This provides:
- ~125x reduction in softmax computation
- Proportional memory reduction for intermediate tensors
- Reduced communication volume (gather K values per rank instead of V/TP)

The trade-off is a slight approximation: probability mass in the tail is ignored. In practice,
with K=1024, this covers >99.9% of the probability mass for typical language models.

---

## 5. Dynamic Loss Balancing

### What

Automatically scales heterogeneous loss terms to compatible magnitudes, eliminating the need
for manual hyperparameter tuning of loss weights.

### Why

Logits KL-divergence and intermediate cosine/MSE losses have vastly different scales. Static
weighting requires expensive hyperparameter search and changes as training progresses.

### How

`LogitsAndIntermediatesLossBalancer.forward()` — `plugins/megatron.py:480-516`

**Step 1: Scale intermediate to logits magnitude**

```python
if intermediate_loss > 0:
    dynamic_scale = logits_loss.detach() / intermediate_loss.detach()
    intermediate_loss_scaled = intermediate_loss * dynamic_scale
```

**Step 2: Scale combined KD loss to original LM loss magnitude** (when `skip_lm_loss=False`):

```python
kd_loss = logits_loss + intermediate_loss_scaled
if kd_loss > 0 and original_loss > 0:
    kd_loss *= original_loss.detach() / kd_loss.detach()
total_loss = original_loss + kd_loss * self._kd_loss_scale
```

**`.detach()` usage**: All scaling factors use `.detach()` to compute scale as a constant,
preventing the scale ratio itself from entering the gradient computation. This means gradients
flow through the original loss values but the scale is treated as a fixed multiplier at each
step.

### Impact

Stable training without manual tuning. The dynamic scaling adapts automatically as loss
magnitudes change during training (e.g., logits loss decreasing faster than intermediate loss).
The only hyperparameter remaining is `kd_loss_scale`, which controls the relative emphasis
between KD and LM losses.

---

## 6. Pipeline Parallelism: Tensor Stacking

### What

Combines student and teacher output tensors into a single tensor for pipeline-parallel (PP)
inter-rank communication, avoiding the need for separate P2P transfers.

### Why

In PP, intermediate activations must be sent between pipeline stages. With distillation, both
student and teacher activations need to be communicated. Sending them separately would double
the number of P2P operations and require modifications to the pipeline scheduler.

### How

`adjust_distillation_model_for_mcore()` — `plugins/megatron.py:586-616`

**Non-last PP stages — concatenation**:

```python
def _forward(self, *args, **kwargs):
    with torch.no_grad():
        teacher_output = self._teacher_model(*args, **kwargs)
    with self.only_student_forward():
        student_output = type(self).forward(self, *args, **kwargs)

    if not parallel_state.is_pipeline_last_stage():
        return torch.cat([student_output, teacher_output], dim=-1)
    else:
        return student_output
```

**Receiving stages — splitting**:

```python
def _set_input_tensor(self, input_tensors):
    teacher_inputs = [t[..., self._tensor_split_idx:] if t is not None else t for t in input_tensors]
    student_inputs = [t[..., :self._tensor_split_idx] if t is not None else t for t in input_tensors]
    type(self).set_input_tensor(self.teacher_model, teacher_inputs)
    type(self).set_input_tensor(self, student_inputs)
```

**Shape adjustment**: `get_tensor_shapes_adjust_fn_for_distillation()`
(`plugins/megatron.py:619-666`) provides a callback that adjusts the expected tensor shapes
in the pipeline scheduler, adding the teacher's hidden dimension to the last dim:

```python
def adjust_tensor_shapes(recv_tensor_shapes, send_tensor_shapes):
    for i, shape in enumerate(recv_tensor_shapes):
        shape = list(shape)
        shape[-1] += teacher_recv_tensor_shapes[0][-1]
        recv_tensor_shapes[i] = tuple(shape)
    # ... same for send_tensor_shapes
```

### Impact

Reuses Megatron's existing PP infrastructure without doubling P2P messages. The concatenated
tensor uses the same communication path as a standard non-distillation forward pass, with only
a larger last dimension. The split on the receiving end is a zero-copy view operation.

---

## 7. Sequence-Parallel Loss Masking

### What

Correctly applies loss masks when the loss tensor is split across TP ranks due to sequence
parallelism.

### Why

With sequence parallelism, the sequence dimension of intermediate activations is split across
TP ranks. Loss masks (which have the full sequence length) must be correspondingly split before
being applied to these partial loss tensors.

### How

`_mask_loss()` — `megatron/post_training/loss_func.py:13-36`

```python
def _mask_loss(output_tensor, loss_mask):
    if isinstance(output_tensor, tuple):
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False

    if is_sequence_parallel:
        idx = parallel_state.get_tensor_model_parallel_rank()
        loss_mask = torch.tensor_split(loss_mask, args.tensor_model_parallel_size, dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    if tp_reduce or is_sequence_parallel:
        torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())
```

The `BaseLoss.post_forward()` method (`plugins/megatron.py:214-219`) packages the loss tensor
with `tp_reduce` and `is_sequence_parallel` flags as a tuple, which `_mask_loss` unpacks.

### Impact

Correct loss computation without gathering full sequence lengths. The `all_reduce` after
masking ensures all TP ranks have the globally-correct scalar loss value for the optimizer.

---

## 8. State Dict Optimization (Minimal Checkpointing)

### What

Excludes the teacher model from saved checkpoints, reducing checkpoint size to student-only.

### Why

The teacher is frozen and never modified. Re-saving it in every checkpoint is wasteful — it
can be loaded from its original checkpoint when resuming.

### How

**Standard state dict**: `distillation_model.py:195-198`

```python
def state_dict(self, *args, **kwargs):
    with self.hide_teacher_model(enable=self._expose_minimal_state_dict):
        return super().state_dict(*args, **kwargs)
```

The `hide_teacher_model()` context manager (`distillation_model.py:144-153`) temporarily
replaces `_teacher_model` with an empty `nn.Module()`, so `state_dict()` sees no teacher
parameters.

**Sharded state dict** (for distributed checkpointing):
`adjust_distillation_model_for_mcore()` patches `sharded_state_dict`
(`plugins/megatron.py:563-568`):

```python
def _sharded_state_dict(self, *args, **kwargs):
    with self.hide_teacher_model():
        return type(self).sharded_state_dict(self, *args, **kwargs)
```

**Default**: `expose_minimal_state_dict=True` is the default in both `DistillationModel.modify()`
(`distillation_model.py:68`) and `KDLossConfig` (`config.py:71-79`).

### Impact

Checkpoint size equals the student model size only. For distillation scenarios where the teacher
is 4-10x larger than the student, this provides substantial I/O and storage savings (e.g., a
70B teacher + 8B student saves only the 8B student).

---

## 9. Projection Layer for Heterogeneous Architectures

### What

A linear projection that maps student hidden-state activations to the teacher's hidden dimension
when the two models have different sizes.

### Why

Intermediate losses (cosine, MSE) require tensors of matching dimensions. When student and
teacher have different `hidden_size` values, student activations must be projected before loss
computation.

### How

`ProjectionLayer` — `plugins/megatron.py:519-552`

```python
class ProjectionLayer(MegatronModule):
    def __init__(self, student_config, teacher_config):
        super().__init__(config=student_config)
        if student_config.hidden_size == teacher_config.hidden_size:
            self._fit = nn.Identity()
        else:
            self._fit = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
            setattr(self._fit.weight, "sequence_parallel", self.config.sequence_parallel)
            setattr(self._fit.bias, "sequence_parallel", self.config.sequence_parallel)
```

**Shared instance**: A single `ProjectionLayer` is created and shared across all intermediate
layer pairs (`plugins/megatron.py:139`):

```python
projection_layer = ProjectionLayer(student_cfg, teacher_cfg)
for entry in cfg.intermediate_layer_pairs:
    ...
    criterion[(student_layer, teacher_layer)] = loss_fn(
        student_cfg, projection_layer=projection_layer
    )
```

**Sequence-parallel gradient reduction**: The `sequence_parallel` attribute on weight and bias
tensors signals Megatron's DDP wrapper to correctly all-reduce gradients for these parameters
across the TP group, matching the behavior of other sequence-parallel parameters in the model.

### Impact

Enables distillation between models of different sizes (e.g., Llama-70B teacher → Llama-8B
student) with minimal parameter overhead — a single linear projection layer regardless of the
number of intermediate loss pairs.

---

## 10. Known Limitations & Their GPU Implications

### `--manual-gc` Incompatibility

**Symptom**: Eventual OOM after many iterations.

**Root cause**: When `mtd.convert()` wraps the model as a `DynamicModule`, it introduces a
small memory allocation per micro-batch forward-backward pass. Python's garbage collector
normally cleans these up. With `--manual-gc`, garbage collection is deferred, and these
allocations accumulate until OOM.

**Guard**: `model_builder.py:310-312` asserts this option is not set:
```python
assert not args.manual_gc, \
    "ModelOpt Distillation currently incompatible with `--manual-gc` option."
```

### `--tp-comm-overlap` Incompatibility

**Symptom**: Incorrect loss values or hangs.

**Root cause**: TP communication overlap reorders collective operations to overlap with
computation. This interferes with the precise ordering required by the distillation forward
hooks and the distributed softmax in `LogitsKLLoss`.

**Guard**: `model_builder.py:313-315` asserts this option is not set.

### Interleaved PP Unsupported

**Symptom**: Assertion failure at model construction.

**Root cause**: Virtual pipeline stages conflict with the tensor concatenation approach used for
PP distillation. The `get_tensor_shapes_adjust_fn_for_distillation()` function returns `None`
when virtual pipeline parallelism is detected (`plugins/megatron.py:629`).

**Guard**: `model_builder.py:198` raises `ValueError` for `vp_stage is not None`, and
`model_builder.py:316-319` asserts `virtual_pipeline_model_parallel_size is None`.

### ~40% Forward Latency Overhead

**Symptom**: Total iteration time significantly longer than the sum of individual student and
teacher forward times.

**Root cause**: A CUDA kernel scheduling issue when running two models sequentially in the same
stream. The teacher and student forward passes are serialized (teacher first, then student),
and the GPU's kernel scheduler introduces overhead between the two passes.

**Documented**: `megatron/post_training/docs/distillation.md:92-93`:
> A CUDA kernel issue is occurring where student's forward latency is severely prolonged
> compared to running student forward without a teacher model.

This is a known hardware/runtime limitation, not a software bug. Potential mitigation would
involve running teacher and student on separate CUDA streams, but this would complicate the
current synchronous hook-based design.
