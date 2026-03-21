# Tensor Parallelism Deep-Dive: Selective TP for Embeddings in Megatron-LM

This guide explains how tensor parallelism (TP) works in Megatron-LM at the lowest level,
and how to selectively apply TP only to `VocabParallelEmbedding` layers — useful when your
model has large per-layer embeddings that dominate GPU memory, but where you want to keep
other layers unsharded.

---

## Part 1: Tensor Parallelism Concepts

### What TP Is and Why It Exists

Tensor parallelism splits individual weight matrices across multiple GPUs. Instead of each
GPU holding a full copy of a layer's parameters (as in data parallelism), each GPU holds
only a **shard** of the weight matrix.

**Why it matters**:
- **Memory**: A 50,000 × 4,096 embedding matrix uses ~400 MB in fp16. With TP=4, each GPU
  holds only ~100 MB.
- **Compute**: Matrix multiplications are distributed — each GPU computes only its portion.
- **Scaling**: Enables layers that are too large for a single GPU's memory.

TP is most effective **within a node** (NVLink interconnect) because it requires
high-bandwidth communication on every forward and backward pass.

### Two Fundamental Sharding Strategies

#### Column-Parallel (split output dimension)

The weight matrix `A` (shape `[input_size, output_size]`) is split along its **column**
(output) dimension:

```
A = [A_1 | A_2 | ... | A_p]   (p = TP world size)

GPU 0 holds A_1 (shape: input_size × output_size/p)
GPU 1 holds A_2
...
```

Each GPU computes `Y_i = X @ A_i`, producing a partial output. The outputs are
**concatenated** (via all-gather) if needed downstream, or kept distributed.

#### Row-Parallel (split input dimension)

The weight matrix `A` is split along its **row** (input) dimension:

```
A = [ A_1 ]   X = [X_1 | X_2 | ... | X_p]
    [ A_2 ]
    [ ... ]
    [ A_p ]
```

Each GPU computes `Y_i = X_i @ A_i` (a partial matrix product). The outputs are
**summed** (via all-reduce) to get the final result, since the full computation is
`Y = X_1 @ A_1 + X_2 @ A_2 + ... + X_p @ A_p`.

### Communication Primitives

| Primitive | Description | When Used |
|-----------|-------------|-----------|
| **All-Reduce** | Sum tensors across all ranks; result on every rank | Row-parallel forward, embedding forward |
| **All-Gather** | Gather shards from all ranks into full tensor | Column-parallel output gathering |
| **Reduce-Scatter** | Reduce (sum) + scatter result across ranks | Sequence parallel gradient sync |

### The f and g Operators

Megatron-LM uses a pair of conjugate operators (`f` and `g`) to handle the **asymmetry**
between forward and backward communication:

**Operator f** (used at column-parallel input):
- Forward: **identity** (copy input to all TP ranks)
- Backward: **all-reduce** (sum gradients from all TP ranks)

**Operator g** (used at row-parallel output):
- Forward: **all-reduce** (sum partial outputs from all TP ranks)
- Backward: **identity** (pass gradients through)

These are implemented as custom `torch.autograd.Function` classes so that PyTorch's
autograd engine automatically applies the correct communication in each direction.

When column-parallel and row-parallel layers are composed (e.g., in an MLP:
`ColumnParallel → activation → RowParallel`), the f/g operators **cancel out** in the
middle, requiring communication only at the boundaries.

---

## Part 2: Low-Level Implementation in Megatron-LM

### `parallel_state.py`: Process Group Management

All parallelism in Megatron-LM starts with process group initialization. The TP group
defines which GPUs share a tensor-parallel partition.

```python
# megatron/core/parallel_state.py:521-526
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    ...
```

The global TP group is stored in `_TENSOR_MODEL_PARALLEL_GROUP` and accessed via:

```python
# megatron/core/parallel_state.py:1330-1336
def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), "tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP
```

Rank and world size within the TP group are queried with
`get_tensor_model_parallel_rank()` and `get_tensor_model_parallel_world_size()`.

**Key insight**: Every TP-aware layer accepts an optional `tp_group` parameter. When
`None`, it falls back to the global TP group. This is the mechanism that enables selective
TP.

### `mappings.py`: Custom Autograd Communication Functions

The communication primitives are implemented as `torch.autograd.Function` subclasses,
making forward/backward communication **asymmetric by design**.

#### `_CopyToModelParallelRegion` (the "f" operator)

```python
# megatron/core/tensor_parallel/mappings.py:197-214
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_                              # Forward: identity (no-op)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None  # Backward: all-reduce
```

In the forward pass, input is simply passed through (identity). In the backward pass,
gradients are all-reduced across the TP group, ensuring every rank gets the aggregated
gradient.

#### `_ReduceFromModelParallelRegion` (the "g" operator)

```python
# megatron/core/tensor_parallel/mappings.py:217-233
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)      # Forward: all-reduce

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None           # Backward: identity (no-op)
```

Forward performs all-reduce to combine partial results from all TP ranks. Backward simply
passes gradients through unchanged.

#### Other Key Operators

| Class | Forward | Backward | Used By |
|-------|---------|----------|---------|
| `_ScatterToModelParallelRegion` | Split along last dim | All-gather along last dim | RowParallel input scatter |
| `_GatherFromModelParallelRegion` | All-gather along last dim | Split along last dim | ColumnParallel output gather |
| `_ScatterToSequenceParallelRegion` | Split along first dim | All-gather along first dim | Sequence parallel scatter |
| `_GatherFromSequenceParallelRegion` | All-gather along first dim | Reduce-scatter along first dim | Sequence parallel gather |
| `_ReduceScatterToSequenceParallelRegion` | Reduce-scatter along first dim | All-gather along first dim | Sequence parallel reduce |

Each operator has a corresponding wrapper function that resolves the TP group:

```python
# megatron/core/tensor_parallel/mappings.py:469-472
def copy_to_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: copy, backward allreduce"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _CopyToModelParallelRegion.apply(input_, group)
```

The `get_tensor_model_parallel_group_if_none(group)` pattern appears throughout: if an
explicit group is provided, use it; otherwise fall back to the global TP group.

### `layers.py`: VocabParallelEmbedding

`VocabParallelEmbedding` splits the vocabulary table across TP ranks. Each rank holds
rows for a contiguous range of token IDs.

#### Initialization

```python
# megatron/core/tensor_parallel/layers.py:189-260
class VocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tp_group = tp_group

        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)

        # Compute this rank's vocab range
        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_pg_rank(self.tp_group),
                get_pg_size(self.tp_group),
            )
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
```

The vocab range is computed by `VocabUtility`:

```python
# megatron/core/tensor_parallel/utils.py:97-121
class VocabUtility:
    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
```

With a vocab of 50,000 and TP=4:
- Rank 0: tokens [0, 12500)
- Rank 1: tokens [12500, 25000)
- Rank 2: tokens [25000, 37500)
- Rank 3: tokens [37500, 50000)

Each rank allocates only its partition: `weight` has shape
`[num_embeddings_per_partition, embedding_dim]`.

#### Forward Pass: Masked Lookup + All-Reduce

```python
# megatron/core/tensor_parallel/layers.py:262-295
def forward(self, input_):
    if self.tp_group.size() > 1:
        # Build the mask: True for tokens NOT on this rank
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Shift to local indices, mask out-of-range to 0
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
    else:
        masked_input = input_

    # Lookup embeddings (only valid for this rank's token range)
    output_parallel = F.embedding(masked_input, self.weight)

    # Zero out embeddings for tokens not on this rank
    if self.tp_group.size() > 1:
        output_parallel[input_mask, :] = 0.0

    if self.reduce_scatter_embeddings:
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(
            output_parallel, group=self.tp_group
        )
    else:
        # All-reduce: sum across TP ranks to get full embeddings
        output = reduce_from_tensor_model_parallel_region(
            output_parallel, group=self.tp_group
        )
    return output
```

**How it works**:
1. Each rank masks out token IDs outside its range, replacing them with index 0
2. Each rank performs a local embedding lookup — for out-of-range tokens, the looked-up
   value is zeroed out
3. All-reduce (sum) across TP ranks combines the results: for each token, exactly one
   rank contributed the real embedding, all others contributed zeros

This is elegant because it requires no explicit routing — the masking + summation achieves
the same effect as looking up the full embedding table.

### `layers.py`: ColumnParallelLinear

Splits the weight matrix along the output (column) dimension.

```python
# megatron/core/tensor_parallel/layers.py:745-813
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        ...
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        ...
        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group, ...)
        world_size = get_pg_size(self.tp_group)
        self.output_size_per_partition = divide(output_size, world_size)
```

Weight shape: `[output_size / TP, input_size]`. Each rank computes a slice of the output.

**Forward data flow** (`layers.py:948-1045`):
1. If not using sequence parallel or explicit expert comm, apply `f` operator
   (`copy_to_tensor_model_parallel_region`) to input
2. Compute `output_parallel = input @ weight.T` (local matmul)
3. If `gather_output=True`, all-gather across TP ranks to produce full output
4. Otherwise, return the partial output (distributed across ranks)

### `layers.py`: RowParallelLinear

Splits the weight matrix along the input (row) dimension.

```python
# megatron/core/tensor_parallel/layers.py:1075-1127
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.
    The linear layer is defined as Y = XA + b.
    A is parallelized along its first dimension and X along its second dimension.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        ...
        input_is_parallel: bool,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        ...
        self.input_size_per_partition = divide(input_size, world_size)
```

Weight shape: `[output_size, input_size / TP]`. Each rank processes a slice of the input.

**Forward data flow** (`layers.py:1232-1288`):
1. If `input_is_parallel=True`, input is already split across TP ranks
2. Compute `output_parallel = input_parallel @ weight.T` (local matmul — partial sum)
3. Apply `g` operator: all-reduce (or reduce-scatter for sequence parallel) to combine
   partial sums

### `random.py`: RNG State Management

TP requires that certain random operations (like dropout) produce **identical** results
across all TP ranks, while others (like weight initialization) produce **different**
results.

```python
# megatron/core/tensor_parallel/random.py:161-168
class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.
    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """
```

Three RNG states are tracked (`random.py:378-429`):
- **default state** (data-parallel seed): Same across TP ranks, different across DP
  groups. Used for dropout in non-TP regions.
- **tensor-model-parallel state**: Different across TP ranks (`seed + 2718 + tp_rank`).
  Used for weight initialization and dropout in TP regions.
- **expert-parallel state**: For MoE expert layers.

The `fork()` context manager (`random.py:242-278`) temporarily switches the CUDA RNG
state:

```python
# megatron/core/tensor_parallel/random.py:242-261
@contextlib.contextmanager
def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
    """Fork the cuda rng state, perform operations, and exit with
    the original state."""
    orig_cuda_rng_state = _get_cuda_rng_state(...)
    _set_cuda_rng_state(self.states_[name], ...)
    try:
        yield
    finally:
        self.states_[name] = _get_cuda_rng_state(...)
        _set_cuda_rng_state(orig_cuda_rng_state, ...)
```

Weight initialization uses the TP-specific RNG:

```python
# megatron/core/tensor_parallel/layers.py:128-140
def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1, is_expert=False):
    set_tensor_model_parallel_attributes(tensor=weight, is_parallel=True, ...)
    if not is_expert:
        with get_cuda_rng_tracker().fork():
            init_method(weight)
```

This ensures that the full (unsharded) weight conceptually gets the same initialization
regardless of the TP configuration.

---

## Part 3: Activating TP Selectively on Embeddings

### How TP Configuration Flows

The TP group propagates through the system via this chain:

```
parallel_state.initialize_model_parallel(tensor_model_parallel_size=N)
    ↓
ProcessGroupCollection.use_mpu_process_groups()
    → pgs.tp = parallel_state.get_tensor_model_parallel_group()
    ↓
GPTModel.__init__(..., pg_collection=pgs)
    → self.embedding = LanguageModelEmbedding(..., tp_group=self.pg_collection.tp)
    → self.decoder = TransformerBlock(..., pg_collection=self.pg_collection)
    → self.output_layer = ColumnParallelLinear(..., tp_group=self.pg_collection.tp)
    ↓
LanguageModelEmbedding.__init__(..., tp_group=tp_group)
    → self.word_embeddings = VocabParallelEmbedding(..., tp_group=self.tp_group)
```

Every TP-aware layer (`VocabParallelEmbedding`, `ColumnParallelLinear`,
`RowParallelLinear`) accepts an explicit `tp_group` parameter. This is the key
extensibility point.

### The `tp_group` Parameter Mechanism

Looking at `VocabParallelEmbedding.__init__` (`layers.py:204-228`):

```python
def __init__(self, ..., tp_group: Optional[torch.distributed.ProcessGroup] = None):
    ...
    self.tp_group = tp_group
    self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)
    # All subsequent operations use self.tp_group for:
    # - Computing vocab range partition
    # - Weight allocation size
    # - All-reduce in forward pass
```

And `LanguageModelEmbedding.__init__` (`language_model_embedding.py:29-63`):

```python
def __init__(self, ..., tp_group: Optional[torch.distributed.ProcessGroup] = None):
    ...
    self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
        ..., tp_group=self.tp_group,
    )
```

The `tp_group` parameter is threaded all the way from `GPTModel` down to the embedding
layer. This means you can provide a **different** process group to embeddings than to the
rest of the model.

### Approach 1: Custom Process Groups (Recommended)

Create two process groups: a real TP group for embeddings, and a trivial (size-1) group
for everything else.

```python
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.models.gpt.gpt_model import GPTModel

# Step 1: Initialize distributed (but NOT model parallel yet)
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()

# Step 2: Initialize standard model parallelism with TP=1
# This means all layers default to no tensor parallelism
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,  # No TP for general layers
)

# Step 3: Create a TP group specifically for embeddings
# Example: 4 GPUs, we want TP=4 for embeddings only
embedding_tp_size = 4
embedding_tp_ranks = list(range(embedding_tp_size))  # [0, 1, 2, 3]
embedding_tp_group = dist.new_group(ranks=embedding_tp_ranks)

# Step 4: Build the process group collection
# Use default groups for everything, but override tp for embeddings
pgs = ProcessGroupCollection.use_mpu_process_groups()

# Step 5: Build the model, passing the embedding TP group
# Option A: Subclass GPTModel to inject the embedding tp_group
class SelectiveTPGPTModel(GPTModel):
    def __init__(self, *args, embedding_tp_group=None, **kwargs):
        self._embedding_tp_group = embedding_tp_group
        super().__init__(*args, **kwargs)

    def _build_embedding(self):
        """Override to use custom TP group for embeddings."""
        from megatron.core.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
                tp_group=self._embedding_tp_group,  # Custom TP group!
            )

model = SelectiveTPGPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=50000,
    max_sequence_length=2048,
    embedding_tp_group=embedding_tp_group,
    pg_collection=pgs,
)
```

With this approach:
- `VocabParallelEmbedding` uses `embedding_tp_group` (size 4) → vocab is split 4 ways
- `ColumnParallelLinear` and `RowParallelLinear` in the transformer use the default TP
  group (size 1) → weights are not split
- Memory savings come from the embedding table being distributed

### Approach 2: Override VocabParallelEmbedding Directly

If you don't want to subclass `GPTModel`, you can construct the embedding layer
separately and inject it:

```python
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)

# Build model normally with TP=1
model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=50000,
    max_sequence_length=2048,
    pg_collection=pgs,
)

# Replace the embedding with one that uses a custom TP group
if model.pre_process:
    model.embedding = LanguageModelEmbedding(
        config=config,
        vocab_size=50000,
        max_sequence_length=2048,
        position_embedding_type='rope',
        tp_group=embedding_tp_group,  # Custom TP group for embeddings
    )
```

This is simpler but more fragile — it bypasses any initialization logic that `GPTModel`
might perform on the embedding.

### Approach 3: Custom ProcessGroupCollection with Different TP

For maximum control, build a custom `ProcessGroupCollection` that carries both the
default TP group and the embedding TP group:

```python
# Create two ProcessGroupCollections
default_pgs = ProcessGroupCollection.use_mpu_process_groups()

# For the embedding, manually construct LanguageModelEmbedding
# with embedding_tp_group, while the model uses default_pgs
# for everything else (TransformerBlock, output_layer, etc.)
```

### Considerations

#### Checkpointing Compatibility

`VocabParallelEmbedding.sharded_state_dict()` (`layers.py:297-316`) uses the `tp_group`
for checkpoint sharding metadata:

```python
def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
    return {
        weight_prefix: make_tp_sharded_tensor_for_checkpoint(
            tensor=state_dict["weight"],
            key=weight_prefix,
            allow_shape_mismatch=True,
            tp_group=self.tp_group,    # Uses the custom TP group
            dp_cp_group=metadata["dp_cp_group"],
        )
    }
```

Because the `tp_group` is stored on the layer itself, checkpoints will correctly record
that the embedding is sharded across the embedding TP group while other layers are
unsharded. Loading a checkpoint with a different TP configuration works because
Megatron-LM's distributed checkpointing uses `allow_shape_mismatch=True` for embeddings.

**Caveat**: If you change `embedding_tp_size` between saving and loading, ensure the
checkpoint loading code uses the same TP group configuration. Megatron's
`dist_checkpointing` handles resharding, but the process groups must be correctly set up.

#### Gradient Synchronization

With selective TP, gradients flow differently for embeddings vs. other layers:

- **Embedding gradients**: The all-reduce in `VocabParallelEmbedding.forward()` handles
  gradient synchronization across the embedding TP group automatically (via the backward
  pass of `reduce_from_tensor_model_parallel_region`).
- **Other layer gradients**: With TP=1 for other layers, there is no TP gradient sync
  needed — data parallelism handles gradient aggregation.
- **Shared embeddings**: If using `share_embeddings_and_output_weights=True`, the output
  layer's `ColumnParallelLinear` must use the **same** TP group as the embedding.
  Otherwise, the shared weight will have inconsistent sharding.

#### Memory Implications

For a model with vocabulary size V and hidden dimension H:

| Configuration | Embedding memory per GPU | Transformer layer memory per GPU |
|---------------|--------------------------|----------------------------------|
| TP=1 (baseline) | `V × H × dtype_size` | Full layer size |
| TP=4 (global) | `V/4 × H × dtype_size` | Layer size / 4 |
| Selective TP=4 (embeddings only) | `V/4 × H × dtype_size` | Full layer size |

Selective TP gives you embedding memory savings without the communication overhead of TP
for every transformer layer.

#### Communication Overhead

- **Embedding all-reduce**: One all-reduce per forward pass per embedding lookup, with
  message size `batch × seq_len × hidden_dim`. This is the same cost as standard TP.
- **No TP overhead for other layers**: Since transformer layers use TP=1, there are no
  all-reduce/all-gather operations for attention or MLP layers.
- **Net effect**: Lower total communication volume compared to global TP, at the cost of
  higher per-GPU compute for non-embedding layers.

#### Process Group Lifetime

Process groups created with `torch.distributed.new_group()` persist until
`torch.distributed.destroy_process_group()` is called. Ensure that:

1. The embedding TP group is created **after** `dist.init_process_group()` but **before**
   model construction
2. All ranks in the embedding TP group participate in the `new_group()` call
3. The group is consistent across model construction, training, and checkpointing

---

## Summary

| Concept | Implementation | File:Line |
|---------|---------------|-----------|
| TP group management | `_TENSOR_MODEL_PARALLEL_GROUP` global | `parallel_state.py:1330` |
| f operator (copy fwd, allreduce bwd) | `_CopyToModelParallelRegion` | `mappings.py:197` |
| g operator (allreduce fwd, copy bwd) | `_ReduceFromModelParallelRegion` | `mappings.py:217` |
| Vocab sharding + masked lookup | `VocabParallelEmbedding` | `layers.py:189` |
| Column-parallel linear | `ColumnParallelLinear` | `layers.py:745` |
| Row-parallel linear | `RowParallelLinear` | `layers.py:1075` |
| RNG state tracking | `CudaRNGStatesTracker` | `random.py:161` |
| Embedding wrapper (accepts tp_group) | `LanguageModelEmbedding` | `language_model_embedding.py:14` |
| Process group collection | `ProcessGroupCollection` | `process_groups_config.py:27` |
| GPT model embedding wiring | `GPTModel.__init__` | `gpt_model.py:147` |

The key takeaway: every TP-aware layer in Megatron-LM accepts an explicit `tp_group`
parameter. By providing a custom process group to embedding layers while leaving other
layers with a trivial (size-1) group, you achieve selective tensor parallelism — reducing
embedding memory across GPUs without the overhead of sharding every layer.
