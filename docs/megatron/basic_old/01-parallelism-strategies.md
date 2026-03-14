# Parallelism Strategies in Megatron-LM

## Table of Contents

1. [Key Takeaways](#key-takeaways)
2. [Introduction](#introduction)
3. [Process Group Management](#process-group-management)
4. [Tensor Parallelism (TP)](#tensor-parallelism-tp)
5. [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
6. [Data Parallelism (DP)](#data-parallelism-dp)
7. [Context Parallelism (CP)](#context-parallelism-cp)
8. [Expert Parallelism (EP)](#expert-parallelism-ep)
9. [Multi-Dimensional Parallelism](#multi-dimensional-parallelism)
10. [NCCL Configuration and Tuning](#nccl-configuration-and-tuning)
11. [Configuration Guidelines](#configuration-guidelines)
12. [Performance Impact](#performance-impact)

---

## Key Takeaways

**Why Parallelism Matters for GPU Utilization**:
- Enables training models too large for single GPU memory
- Minimizes GPU idle time through strategic work distribution
- Achieves near-linear scaling to 1000+ GPUs
- Allows flexible trade-offs between memory, communication, and computation

**Core Strategy**:
Megatron combines **5 orthogonal parallelism dimensions** (TP, PP, DP, CP, EP) that can be configured independently. The key to high GPU utilization is balancing these dimensions to minimize:
1. **Pipeline bubbles** (idle time in PP)
2. **Communication overhead** (all-reduce, all-gather, reduce-scatter)
3. **Load imbalance** (uneven work distribution across GPUs)

**Critical Numbers**:
- **15+ specialized process groups** for fine-grained communication control
- **Near-linear scaling** demonstrated up to 1024 GPUs for large models
- **Communication can be 80-90% hidden** through overlap techniques
- **Total GPUs = TP × PP × CP × EP × DP**

---

## Introduction

### The Scaling Challenge

Training modern large language models presents a fundamental challenge: **a single GPU cannot hold the entire model, optimizer states, and activations in memory**. Even with the latest H100 GPUs (80GB HBM3), models like LLaMA-3.1 405B or DeepSeek-V3 671B require distributed training across hundreds or thousands of GPUs.

However, simply adding more GPUs doesn't automatically improve GPU utilization. Without careful parallelization strategy, GPUs can spend significant time:
- **Idle** (waiting for data or synchronization)
- **Communicating** (transferring weights/gradients between GPUs)
- **Load-imbalanced** (some GPUs finish early, others lag)

### Megatron's Multi-Dimensional Approach

Megatron-LM solves this through a sophisticated **multi-dimensional parallelism** framework that enables:

1. **Memory Distribution**: Split model across GPUs to fit in memory
2. **Computation Distribution**: Parallelize work to reduce training time
3. **Communication Optimization**: Overlap communication with computation
4. **Flexibility**: Adapt strategy to model architecture and hardware topology

The genius of Megatron's design is that **each parallelism dimension serves a specific purpose** and they compose cleanly:

```
Total GPUs = TP × PP × CP × EP × DP

Example: 1024 GPUs for DeepSeek-V3 671B
TP=2, PP=16, EP=64, CP=1, DP=varies
→ 2 × 16 × 64 × 1 × DP = 1024
→ DP = 0.5 (gradient accumulation across nodes)
```

---

## Process Group Management

### Overview

At the core of Megatron's parallelism infrastructure is the **process group management system** implemented in `parallel_state.py:1-2687`. This module creates and manages **15+ specialized NCCL process groups**, each optimized for different communication patterns.

### Why Multiple Process Groups?

Each parallelism dimension requires different communication patterns:
- **TP**: Frequent small all-reduce operations (intra-node, low latency critical)
- **PP**: Point-to-point sends/receives (inter-node, large messages)
- **DP**: Infrequent large all-reduce of gradients (can tolerate higher latency)
- **CP**: Ring all-to-all for attention (sequential dependency)
- **EP**: All-to-all for token routing (irregular message sizes)

**Using separate process groups allows**:
1. **Independent NCCL tuning** per communication pattern
2. **Concurrent communication** across different dimensions
3. **Hierarchical network topology awareness** (NVLink vs InfiniBand)

### Process Group Hierarchy

From `parallel_state.py:27-100`, Megatron creates the following global process groups:

```python
# Core parallelism groups
_TENSOR_MODEL_PARALLEL_GROUP = None           # TP group
_PIPELINE_MODEL_PARALLEL_GROUP = None         # PP group
_DATA_PARALLEL_GROUP = None                   # DP group
_DATA_PARALLEL_GROUP_GLOO = None              # DP group (Gloo backend for CPU)

# Combined groups
_MODEL_PARALLEL_GROUP = None                  # TP + PP combined
_TENSOR_AND_DATA_PARALLEL_GROUP = None        # TP + DP (for FP8, MoE)

# Expert parallelism groups (MoE)
_EXPERT_MODEL_PARALLEL_GROUP = None           # Expert parallelism
_EXPERT_TENSOR_PARALLEL_GROUP = None          # TP within experts
_EXPERT_DATA_PARALLEL_GROUP = None            # DP for experts
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None # Combined expert groups

# Specialized groups
_EMBEDDING_GROUP = None                        # Embedding synchronization
_POSITION_EMBEDDING_GROUP = None               # Position embedding sync
```

### NCCL Configuration Per Group

Megatron supports **per-group NCCL configuration** for performance tuning:

```python
# From parallel_state.py:141-173
nccl_options = torch.distributed.ProcessGroupNCCL.Options(
    is_high_priority_stream=nccl_comm_cfgs[pg_name].get("is_high_priority_stream", False)
)

# Network selection (InfiniBand vs Socket)
if "net_name" in nccl_comm_cfgs[pg_name]:
    nccl_options.config.net_name = nccl_comm_cfgs[pg_name]["net_name"]

# NCCL kernel tuning
if "cga_cluster_size" in nccl_comm_cfgs[pg_name]:
    nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name]["cga_cluster_size"]
```

**Key configuration parameters**:
- `is_high_priority_stream`: Critical path communications get higher priority
- `net_name`: Select specific network interface (e.g., "IB" for InfiniBand)
- `cga_cluster_size`, `max_ctas`, `min_ctas`: CUDA kernel tuning for NCCL

### SHARP Acceleration

For **data-parallel groups**, Megatron enables **NVIDIA SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol):

```python
# From parallel_state.py - SHARP enables in-network reduction
# Reduces CPU overhead and improves bandwidth utilization
# Only applied to first DP group for hardware compatibility
```

**SHARP benefits**:
- **Hardware-accelerated all-reduce** in the switch fabric
- **50-70% reduction in CPU overhead** for gradient synchronization
- **Higher effective bandwidth** by offloading aggregation to network

---

## Tensor Parallelism (TP)

### Concept

**Tensor Parallelism** splits individual layers horizontally across GPUs. Each GPU holds a **shard of the weight matrices** and performs computation on that shard.

**Key insight**: Matrix multiplication can be split along either dimension:
- **Column-parallel**: Split output dimension, requires all-reduce of outputs
- **Row-parallel**: Split input dimension, requires all-reduce of inputs

### Implementation

Implemented in `tensor_parallel/layers.py:1-1425`, Megatron provides:
- `ColumnParallelLinear`: Splits weight columns across TP ranks
- `RowParallelLinear`: Splits weight rows across TP ranks
- `VocabParallelEmbedding`: Splits vocabulary across TP ranks

### Implementation Deep Dive

Let's examine exactly how weights and tensors are split and communicated in ColumnParallelLinear and RowParallelLinear.

#### ColumnParallelLinear: Weight Sharding and Data Flow

**Weight Initialization** (`layers.py:834-876`):

```python
# megatron/core/tensor_parallel/layers.py:834-876
# Original weight shape: [output_size, input_size]
# Each TP rank stores: [output_size_per_partition, input_size]

world_size = tp_group.size()  # e.g., 4 GPUs
self.output_size_per_partition = divide(output_size, world_size)
# If output_size=4096, TP=4: output_size_per_partition=1024

self.weight = Parameter(
    torch.empty(
        self.output_size_per_partition,  # Sharded dimension!
        self.input_size,                  # Replicated dimension
        device=torch.cuda.current_device(),
        dtype=config.params_dtype,
    )
)
```

**Weight distribution across ranks**:
```
Full weight matrix A: [4096, 1024]

With TP=4:
Rank 0: A[0:1024, :]    = A_0  (first 1024 output features)
Rank 1: A[1024:2048, :] = A_1  (next 1024 output features)
Rank 2: A[2048:3072, :] = A_2
Rank 3: A[3072:4096, :] = A_3

Memory: Each GPU stores 1/4 of output dimension
```

**Forward Pass WITHOUT Sequence Parallelism** (`layers.py:991-1043`):

```python
# megatron/core/tensor_parallel/layers.py:991-1043

# Input shape: [seq_len, batch, hidden_in]
# Example: [2048, 4, 1024]

# Step 1: Broadcast input to all TP ranks (if not already parallel)
if not (self.allreduce_dgrad or self.sequence_parallel):
    input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)
else:
    input_parallel = input_  # Input already on all ranks

# Step 2: Local matrix multiplication
# Each rank computes: output_i = input @ weight_i.T
output_parallel = torch.matmul(input_parallel, self.weight.t())

# Input:  [2048, 4, 1024]  (replicated on all ranks)
# Weight: [1024, 1024]     (rank 0: first 1024 outputs)
#         [1024, 1024]     (rank 1: next 1024 outputs)
#         ...
# Output: [2048, 4, 1024]  (rank 0: first 1024 features)
#         [2048, 4, 1024]  (rank 1: next 1024 features)
#         ...

# Step 3: Optionally gather outputs
if self.gather_output:
    # All-gather across TP ranks to get full output
    output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
    # Result: [2048, 4, 4096]  (concatenated along last dim)
else:
    output = output_parallel  # Keep partitioned
    # Result: [2048, 4, 1024] on each rank
```

**Forward Pass WITH Sequence Parallelism** (`layers.py:470-483`):

```python
# megatron/core/tensor_parallel/layers.py:470-483

# Input shape: [seq_len/tp_size, batch, hidden_in]
# Example with TP=4: [512, 4, 1024] on each rank

if sequence_parallel:
    # Step 1: All-gather input along sequence dimension
    dim_size = list(input.size())
    dim_size[0] = dim_size[0] * tp_group.size()  # 512 * 4 = 2048

    all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
    dist_all_gather_func(all_gather_buffer, input, group=tp_group)
    total_input = all_gather_buffer
    # Result: [2048, 4, 1024]  (full sequence on all ranks)

# Step 2: Matrix multiplication (same as before)
output = torch.matmul(total_input, weight.t())
# Result: [2048, 4, 1024] on each rank (partitioned output dimension)

# Note: gather_output typically False with sequence parallelism
```

**Backward Pass** (`layers.py:487-572`):

```python
# megatron/core/tensor_parallel/layers.py:487-572

def backward(ctx, grad_output):
    # grad_output shape: [seq_len, batch, output_per_partition]
    # Example: [2048, 4, 1024]

    input, weight = ctx.saved_tensors

    # Step 1: Compute input gradient
    # grad_input = grad_output @ weight
    grad_input = grad_output.matmul(weight)
    # Shape: [2048, 4, 1024] (input dimension)

    # Step 2: Communication for input gradients
    if ctx.allreduce_dgrad:
        # WITHOUT sequence parallelism:
        # All-reduce to sum gradients from all TP ranks (async!)
        handle = torch.distributed.all_reduce(
            grad_input, group=tp_group, async_op=True
        )
        # Each rank sent partial grad from its output partition
        # Need to sum them all to get true grad_input

    elif ctx.sequence_parallel:
        # WITH sequence parallelism:
        # Reduce-scatter to partition gradients back along sequence
        sub_grad_input = torch.empty(
            [seq_len/tp_size, batch, hidden],  # Partitioned shape
            dtype=input.dtype,
            device=torch.cuda.current_device()
        )
        handle = dist_reduce_scatter_func(
            sub_grad_input,   # Output: [512, 4, 1024]
            grad_input,       # Input:  [2048, 4, 1024]
            group=tp_group,
            async_op=True
        )
        # Result: Each rank gets its portion of sequence dimension

    # Step 3: Compute weight gradient (if needed)
    if ctx.sequence_parallel:
        # Re-gather input for weight gradient computation
        all_gather_buffer = get_global_memory_buffer().get_tensor(
            dim_size, input.dtype, "mpu"
        )
        handle_gather = dist_all_gather_func(
            all_gather_buffer, input, group=tp_group, async_op=True
        )
        handle_gather.wait()  # Wait for completion
        total_input = all_gather_buffer
    else:
        total_input = input

    # Compute weight gradient
    if ctx.gradient_accumulation_fusion:
        # Fused gradient accumulation in FP32
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input, grad_output, weight.main_grad
        )
        # Directly accumulates: weight.main_grad += grad_output.T @ total_input
    else:
        grad_weight = torch.matmul(grad_output.t(), total_input)

    return grad_input, grad_weight, ...
```

**Communication Pattern Summary for ColumnParallelLinear**:

```
Forward (no SP):
  Input:  Broadcast to all ranks (or already replicated)
  Compute: Each rank → partial output
  Output: Optionally all-gather (usually not in transformers)

Forward (with SP):
  Input:  All-gather along sequence → full sequence
  Compute: Each rank → partial output
  Output: Keep partitioned

Backward (no SP):
  grad_input:  All-reduce (sum contributions from all ranks)
  grad_weight: Computed locally

Backward (with SP):
  grad_input:  Reduce-scatter along sequence
  grad_weight: All-gather input first, then compute
```

#### RowParallelLinear: Weight Sharding and Data Flow

**Weight Initialization** (`layers.py:1148-1201`):

```python
# megatron/core/tensor_parallel/layers.py:1148-1201
# Original weight shape: [output_size, input_size]
# Each TP rank stores: [output_size, input_size_per_partition]

world_size = tp_group.size()
self.input_size_per_partition = divide(input_size, world_size)
# If input_size=4096, TP=4: input_size_per_partition=1024

self.weight = Parameter(
    torch.empty(
        self.output_size,                 # Full output dimension!
        self.input_size_per_partition,    # Sharded dimension!
        device=torch.cuda.current_device(),
        dtype=config.params_dtype,
    )
)
```

**Weight distribution across ranks**:
```
Full weight matrix A: [1024, 4096]

With TP=4:
Rank 0: A[:, 0:1024]    = A_0  (first 1024 input features)
Rank 1: A[:, 1024:2048] = A_1  (next 1024 input features)
Rank 2: A[:, 2048:3072] = A_2
Rank 3: A[:, 3072:4096] = A_3

Memory: Each GPU stores 1/4 of input dimension
```

**Forward Pass** (`layers.py:1225-1281`):

```python
# megatron/core/tensor_parallel/layers.py:1225-1281

# Input shape depends on input_is_parallel flag

# Step 1: Scatter input if not already parallel
if self.input_is_parallel:
    input_parallel = input_
    # Already partitioned: [seq_len, batch, hidden_per_partition]
    # Example: [2048, 4, 1024]
else:
    # Need to split input across last dimension
    input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)
    # Input:  [2048, 4, 4096]  (full)
    # Output: [2048, 4, 1024]  (partitioned on each rank)

# Step 2: Local matrix multiplication
output_parallel = torch.matmul(input_parallel, self.weight.t())

# Input:  [2048, 4, 1024]  (rank 0: first 1024 features)
# Weight: [1024, 1024]     (rank 0: weight for first 1024 inputs)
# Output: [2048, 4, 1024]  (partial result)

# Step 3: Sum partial results across TP ranks
if self.sequence_parallel:
    # Reduce-scatter along sequence dimension
    output_ = reduce_scatter_to_sequence_parallel_region(
        output_parallel, group=self.tp_group
    )
    # Input:  [2048, 4, 1024] on each rank
    # Output: [512, 4, 1024] on each rank (reduced + scattered)
else:
    # All-reduce to sum partial results
    output_ = reduce_from_tensor_model_parallel_region(
        output_parallel, group=self.tp_group
    )
    # Input:  [2048, 4, 1024] on each rank (partial)
    # Output: [2048, 4, 1024] on each rank (summed across all ranks)
```

**Backward Pass**:

```python
# Backward is simpler because input is already partitioned

def backward(ctx, grad_output):
    # grad_output shape: [seq_len, batch, output_size]
    # Example: [2048, 4, 1024]  (full output dimension on each rank)

    input, weight = ctx.saved_tensors
    # input: [2048, 4, input_per_partition] = [2048, 4, 1024]

    # Step 1: Compute input gradient
    grad_input = grad_output.matmul(weight)
    # Result: [2048, 4, input_per_partition] = [2048, 4, 1024]
    # Already partitioned correctly! No communication needed.

    # Step 2: Compute weight gradient
    grad_weight = torch.matmul(grad_output.t(), input)
    # Result: [1024, 1024] (matches weight shape)

    return grad_input, grad_weight, ...
```

**Communication Pattern Summary for RowParallelLinear**:

```
Forward (no SP):
  Input:  Scatter across last dim (if not already parallel)
  Compute: Each rank → partial output
  Output: All-reduce (sum partial results)

Forward (with SP):
  Input:  Already partitioned (from previous layer's reduce-scatter)
  Compute: Each rank → partial output
  Output: Reduce-scatter along sequence

Backward:
  grad_input:  No communication (already partitioned correctly)
  grad_weight: Computed locally (no communication)
```

#### Memory Layout Example

For a 2-layer MLP with TP=4:

```
Layer 1: ColumnParallelLinear  [1024 → 4096]
  Input:  [2048, 4, 1024]  (replicated on all 4 GPUs)
  Weight: [1024, 1024]     (each GPU has 1/4 of output dim)
  Output: [2048, 4, 1024]  (partitioned)

  Memory per GPU:
    - Weights: 1024 × 1024 × 2 bytes = 2 MB
    - Activations: 2048 × 4 × 1024 × 2 bytes = 16 MB
    Total: ~18 MB per GPU (vs 72 MB if not sharded)

Layer 2: RowParallelLinear  [4096 → 1024]
  Input:  [2048, 4, 1024]  (partitioned, from Layer 1)
  Weight: [1024, 1024]     (each GPU has 1/4 of input dim)
  Output: [2048, 4, 1024]  (replicated after all-reduce)

  Memory per GPU:
    - Weights: 1024 × 1024 × 2 bytes = 2 MB
    - Activations: 2048 × 4 × 1024 × 2 bytes = 16 MB
    Total: ~18 MB per GPU

Total for 2-layer MLP with TP=4:
  - Weights: 4 MB per GPU (vs 16 MB without TP)
  - Activations: 32 MB per GPU
  - 4× memory reduction for weights!
```

#### Sequence Parallelism: Detailed Communication

**All-Gather Operation** (`mappings.py:114-150`):

```python
# megatron/core/tensor_parallel/mappings.py:114-150
def _gather_along_first_dim(input_, group):
    """Gather tensors and concatenate along the first dimension."""

    world_size = group.size()  # e.g., 4

    # Input: [seq_len/4, batch, hidden] = [512, 4, 1024]
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size  # 512 * 4 = 2048

    # Allocate output buffer
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    # Output: [2048, 4, 1024]

    # All-gather: each rank contributes its portion
    dist_all_gather_func(output, input_.contiguous(), group=group)

    # Result layout in memory:
    # output[0:512, :, :]     ← from rank 0
    # output[512:1024, :, :]  ← from rank 1
    # output[1024:1536, :, :] ← from rank 2
    # output[1536:2048, :, :] ← from rank 3

    return output
```

**Reduce-Scatter Operation** (`mappings.py:156-176`):

```python
# megatron/core/tensor_parallel/mappings.py:156-176
def _reduce_scatter_along_first_dim(input_, group):
    """Reduce-scatter tensors along the first dimension."""

    world_size = group.size()  # e.g., 4
    rank = group.rank()

    # Input: [2048, 4, 1024]  (each rank has full tensor)
    dim_size = list(input_.size())
    local_dim_size = dim_size[0] // world_size  # 2048 // 4 = 512

    # Allocate output buffer for local portion
    output = torch.empty(
        [local_dim_size] + dim_size[1:],
        dtype=input_.dtype,
        device=torch.cuda.current_device()
    )
    # Output: [512, 4, 1024]

    # Reduce-scatter: sum across ranks and scatter results
    dist_reduce_scatter_func(output, input_, group=group)

    # Each rank receives:
    # rank 0: sum(all_ranks)[0:512, :, :]
    # rank 1: sum(all_ranks)[512:1024, :, :]
    # rank 2: sum(all_ranks)[1024:1536, :, :]
    # rank 3: sum(all_ranks)[1536:2048, :, :]

    return output
```

**Memory Buffer Management**:

Megatron uses a **global memory buffer** to avoid repeated allocations:

```python
# megatron/core/parallel_state.py
all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
```

This pre-allocates memory pools and reuses them across operations, avoiding:
- Memory allocation overhead
- Memory fragmentation
- CUDA allocation slowdowns

### Column-Parallel Linear Layer

```python
# megatron/core/tensor_parallel/layers.py:~300-400
class ColumnParallelLinear:
    """
    Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Each GPU computes: Y_i = X A_i
    Then concatenate results: Y = [Y_1, ..., Y_p]
    """

    def forward(self, input_):
        # No communication needed in forward pass for column-parallel
        # Each rank computes its partition independently
        output = F.linear(input_, self.weight)

        if self.bias is not None:
            output = output + self.bias

        return output
```

**Communication**:
- **Forward**: None (outputs are concatenated along last dim)
- **Backward**: All-reduce of input gradients

### Row-Parallel Linear Layer

```python
# megatron/core/tensor_parallel/layers.py:~500-600
class RowParallelLinear:
    """
    Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X is split along last dimension:

    X = [X_1, ..., X_p] and A = [A_1
                                  ...
                                  A_p]

    Each GPU computes: Y_i = X_i A_i
    Then sum results: Y = sum(Y_i) via all-reduce
    """

    def forward(self, input_):
        # Each rank has partial input
        output = F.linear(input_, self.weight)

        # All-reduce to sum partial outputs
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                output,
                group=get_tensor_model_parallel_group()
            )

        if self.bias is not None:
            output = output + self.bias

        return output
```

**Communication**:
- **Forward**: All-reduce of outputs
- **Backward**: None (input gradients are already partitioned)

### Async All-Reduce for Overlap

Critical optimization from `layers.py:535-536`:

```python
# megatron/core/tensor_parallel/layers.py:535-536
if ctx.allreduce_dgrad:
    # Asynchronous all-reduce for communication-computation overlap
    handle = torch.distributed.all_reduce(
        grad_input,
        group=tp_group,
        async_op=True  # <-- Non-blocking!
    )
```

**Why async?** The all-reduce can execute concurrently with subsequent backward computations, hiding communication latency.

### Sequence Parallelism

**Problem**: Tensor parallelism only splits weight matrices, but **activations are replicated** across TP ranks. For transformers, activations scale with sequence length, consuming significant memory.

**Solution**: **Sequence Parallelism** partitions activations along the sequence dimension.

```python
# Conceptual implementation
# Input shape: [seq_len, batch, hidden]
# With SP: Each TP rank holds [seq_len/tp_size, batch, hidden]

# In dropout, LayerNorm: operate on partitioned sequence
# Before TP linear: all-gather sequence → [seq_len, batch, hidden/tp_size]
# After TP linear: reduce-scatter sequence → [seq_len/tp_size, batch, hidden]
```

**Memory savings**: ~(TP_size - 1) / TP_size of activation memory
- TP=4: 75% activation memory savings
- TP=8: 87.5% activation memory savings

**Communication**:
- All-gather before TP operations (overlapped with previous computation)
- Reduce-scatter after TP operations (overlapped with next computation)

See `layers.py:514-520` for overlap implementation:

```python
# megatron/core/tensor_parallel/layers.py:514-520
handle = dist_all_gather_func(
    all_gather_buffer, input, group=tp_group, async_op=True
)
# Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
# gather is scheduled before the input gradient computation
```

### Gradient Accumulation Fusion

From `layers.py:553-571`, Megatron fuses gradient accumulation directly into the all-reduce operation:

```python
# megatron/core/tensor_parallel/layers.py:553-571
if ctx.gradient_accumulation_fusion:
    # Directly accumulate gradients in FP32 buffer
    # Avoids intermediate copy: grad_input → grad_input_fp32 → buffer
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
        input, grad_output, grad_weight_buffer
    )
else:
    # Standard path: compute gradient then copy
    grad_weight = torch.matmul(grad_output.t(), input)
```

**Benefit**: Eliminates memory copies, directly accumulates in higher precision buffer.

### TP Configuration Guidelines

**When to use TP**:
- Model doesn't fit in single GPU memory
- Intra-node parallelism (TP works best within a node over NVLink)
- Typically TP ∈ {1, 2, 4, 8} matching GPUs per node

**Constraints**:
- `TP must divide num_attention_heads` (each head can't be split further)
- `TP must divide FFN_hidden_size` for MLP layers
- Use sequence parallelism with TP > 1 for memory efficiency

**Example**:
```bash
# LLaMA-3 70B on 8x H100 node
--tensor-model-parallel-size 8
--sequence-parallel  # Highly recommended with TP > 1
```

---

## Pipeline Parallelism (PP)

### Concept

**Pipeline Parallelism** splits the model **vertically** (across layers) and assigns each partition to a different GPU. Each GPU is a **pipeline stage** that processes its assigned layers sequentially.

**Key challenge**: Naive pipelining creates **bubbles** (idle time) at the beginning and end of each batch.

### Implementation

Pipeline parallelism is managed through `pipeline_parallel/schedules.py:1-2800`, which implements several sophisticated scheduling algorithms to minimize bubbles.

### Pipeline Stages

```
Pipeline Stage 0: Layers 0-7     (GPU 0)
Pipeline Stage 1: Layers 8-15    (GPU 1)
Pipeline Stage 2: Layers 16-23   (GPU 2)
Pipeline Stage 3: Layers 24-31   (GPU 3)

Data flows: GPU 0 → GPU 1 → GPU 2 → GPU 3
Gradients flow: GPU 3 → GPU 2 → GPU 1 → GPU 0
```

### Communication Pattern

**Point-to-point (P2P) send/recv** between adjacent pipeline stages:

```python
# Forward pass: send activations to next stage
if not is_pipeline_last_stage():
    send_forward(output_tensor, pipeline_parallel_next_rank)

# Forward pass: receive activations from previous stage
if not is_pipeline_first_stage():
    input_tensor = recv_forward(pipeline_parallel_prev_rank)

# Backward pass: send gradient to previous stage
if not is_pipeline_first_stage():
    send_backward(input_gradient, pipeline_parallel_prev_rank)

# Backward pass: receive gradient from next stage
if not is_pipeline_last_stage():
    output_gradient = recv_backward(pipeline_parallel_next_rank)
```

### Implementation Deep Dive

Let's examine exactly how activations and gradients flow through pipeline stages, and how microbatches are managed.

#### P2P Communication Implementation

**From `p2p_communication.py:1-300`**, Megatron implements point-to-point send/receive primitives:

```python
# megatron/core/pipeline_parallel/p2p_communication.py

def send_forward(output_tensor, config):
    """Send activation tensor to next pipeline stage."""
    if not isinstance(output_tensor, torch.Tensor):
        raise RuntimeError("output_tensor must be a torch.Tensor")

    # Get next rank in pipeline
    next_rank = parallel_state.get_pipeline_model_parallel_next_rank()

    # Send tensor shape first (for dynamic shapes)
    # Then send actual tensor data
    torch.distributed.send(output_tensor, next_rank)

def recv_forward(tensor_shape, config):
    """Receive activation tensor from previous pipeline stage."""
    prev_rank = parallel_state.get_pipeline_model_parallel_prev_rank()

    # Allocate buffer for incoming tensor
    input_tensor = torch.empty(
        tensor_shape,
        dtype=config.pipeline_dtype,
        device=torch.cuda.current_device()
    )

    # Receive tensor data
    torch.distributed.recv(input_tensor, prev_rank)

    return input_tensor
```

**Actual tensor transfer**:
```
Stage 0 → Stage 1:
  Tensor shape: [micro_batch_size, seq_len, hidden_size]
  Example: [2, 2048, 4096] = 2 × 2048 × 4096 × 2 bytes = 32 MB

  Transfer time over NVLink (900 GB/s): ~0.036 ms
  Transfer time over InfiniBand (50 GB/s): ~0.64 ms
```

#### Microbatch Management in 1F1B

**How microbatches are tracked** (`schedules.py:~400-600`):

```python
# megatron/core/pipeline_parallel/schedules.py

def forward_backward_no_pipelining(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    ...
):
    """
    1F1B schedule implementation.

    Key data structures:
    - input_tensors: List of activation tensors (one per microbatch in flight)
    - output_tensors: List of output tensors from forward
    - losses_reduced: Accumulated losses from all microbatches
    """

    # Initialize storage for microbatches in flight
    input_tensors = []
    output_tensors = []
    forward_data_store = []

    # Calculate warmup microbatches
    # Warmup = number of stages "ahead" of this rank
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )

    # Example for PP=4:
    # Rank 0: warmup = 4 - 0 - 1 = 3 microbatches
    # Rank 1: warmup = 4 - 1 - 1 = 2 microbatches
    # Rank 2: warmup = 4 - 2 - 1 = 1 microbatch
    # Rank 3: warmup = 4 - 3 - 1 = 0 microbatches
```

**1F1B Execution Flow** with detailed microbatch tracking:

```python
# Phase 1: Warmup (fill pipeline)
for i in range(num_warmup_microbatches):
    # Receive input from previous stage (if not first stage)
    if not is_pipeline_first_stage():
        input_tensor = recv_forward(...)
        input_tensors.append(input_tensor)
    else:
        input_tensor = None

    # Forward pass for microbatch i
    output_tensor = forward_step(
        forward_step_func,
        data_iterator[i],
        model,
        input_tensor,
        ...
    )

    # Save output for backward pass later
    output_tensors.append(output_tensor)

    # Send output to next stage (if not last stage)
    if not is_pipeline_last_stage():
        send_forward(output_tensor, ...)

# Phase 2: Steady state (1 forward + 1 backward per iteration)
for i in range(num_warmup_microbatches, num_microbatches):
    # Save current microbatch index for backward
    last_iteration = (i == num_microbatches - 1)

    # FORWARD for microbatch i
    if not is_pipeline_first_stage():
        input_tensor = recv_forward(...)
        input_tensors.append(input_tensor)
    else:
        input_tensor = None

    output_tensor = forward_step(...)
    output_tensors.append(output_tensor)

    if not is_pipeline_last_stage():
        send_forward(output_tensor, ...)

    # BACKWARD for microbatch (i - num_warmup_microbatches)
    backward_microbatch_idx = i - num_warmup_microbatches

    # Receive output gradient from next stage
    if not is_pipeline_last_stage():
        output_tensor_grad = recv_backward(...)
    else:
        output_tensor_grad = None

    # Backward pass
    input_tensor_grad = backward_step(
        input_tensors[backward_microbatch_idx],
        output_tensors[backward_microbatch_idx],
        output_tensor_grad,
        ...
    )

    # Send input gradient to previous stage
    if not is_pipeline_first_stage():
        send_backward(input_tensor_grad, ...)

# Phase 3: Cooldown (drain pipeline with remaining backwards)
for i in range(num_warmup_microbatches):
    backward_microbatch_idx = num_microbatches - num_warmup_microbatches + i

    # Receive and send gradients
    if not is_pipeline_last_stage():
        output_tensor_grad = recv_backward(...)

    input_tensor_grad = backward_step(...)

    if not is_pipeline_first_stage():
        send_backward(input_tensor_grad, ...)
```

**Memory tracking during 1F1B** for Rank 1 with PP=4, M=8:

```
Time →  Microbatch operations           Input tensors   Output tensors
────────────────────────────────────────────────────────────────────────
0:      F0 (warmup)                     [T0]            [O0]
1:      F1 (warmup)                     [T0,T1]         [O0,O1]
2:      F2, B0 (steady state)           [T0,T1,T2]      [O0,O1,O2]
3:      F3, B1                          [T0,T1,T2,T3]   [O0,O1,O2,O3]  ← Peak!
4:      F4, B2                          [T1,T2,T3,T4]   [O1,O2,O3,O4]
5:      F5, B3                          [T2,T3,T4,T5]   [O2,O3,O4,O5]
6:      F6, B4                          [T3,T4,T5,T6]   [O3,O4,O5,O6]
7:      F7, B5                          [T4,T5,T6,T7]   [O4,O5,O6,O7]
8:      B6 (cooldown)                   [T5,T6,T7]      [O5,O6,O7]
9:      B7                              [T6,T7]         [O6,O7]

Peak activation memory: 3 microbatches worth
(vs 8 microbatches for naive GPipe!)
```

#### Interleaved Pipeline (Virtual Pipeline)

**How virtual stages work** (`schedules.py:~1200-1500`):

```python
# With virtual_pipeline_model_parallel_size=2, model is chunked:
# Original: 32 layers on 4 GPUs → 8 layers per GPU
# Interleaved: 32 layers split into 8 chunks → 4 layers per chunk

# Rank 0 holds: chunks [0, 4] = layers [0-3, 16-19]
# Rank 1 holds: chunks [1, 5] = layers [4-7, 20-23]
# Rank 2 holds: chunks [2, 6] = layers [8-11, 24-27]
# Rank 3 holds: chunks [3, 7] = layers [12-15, 28-31]

def get_model_chunk_id(microbatch_id, forward):
    """Determine which model chunk to use for this microbatch."""
    num_model_chunks = get_num_model_chunks()
    microbatch_id_in_group = microbatch_id % (2 * num_model_chunks)

    if forward:
        model_chunk_id = microbatch_id_in_group // 2
    else:
        model_chunk_id = num_model_chunks - (microbatch_id_in_group // 2) - 1

    return model_chunk_id
```

**Interleaved schedule execution** (simplified):

```
PP=4, VP=2, M=8 microbatches

Rank 0 (chunks 0 and 4):
  Time →
  [F0_c0][F1_c0][F0_c4][F1_c4][B0_c0][F2_c0][B1_c0][F2_c4]...

  Notation: F0_c0 = Forward microbatch 0, chunk 0
```

The interleaving reduces bubbles because while waiting for one chunk's activations, the GPU works on another chunk!

#### Tensor Shape Propagation

**Dynamic shape handling** for variable sequence lengths:

```python
# Each pipeline stage may change tensor shapes
# Example: First stage (embedding) → Last stage (loss)

# Stage 0 (embedding):
input_ids: [micro_batch, seq_len] = [2, 2048]
output: [micro_batch, seq_len, hidden] = [2, 2048, 4096]

# Stage 1-2 (transformer layers):
input: [2, 2048, 4096]
output: [2, 2048, 4096]  # Same shape

# Stage 3 (final layer + loss):
input: [2, 2048, 4096]
output: scalar loss value

# Shapes must be communicated for each P2P transfer!
```

**Shape communication** (from `p2p_communication.py`):

```python
def _communicate_shapes(tensor_send_next, recv_prev, config):
    """Exchange tensor shapes before transferring data."""
    # Send shape as small metadata tensor
    if tensor_send_next is not None:
        shape_tensor = torch.LongTensor(list(tensor_send_next.shape))
        torch.distributed.send(shape_tensor, next_rank)

    if recv_prev:
        shape_tensor = torch.LongTensor([3])  # Max 3D tensors
        torch.distributed.recv(shape_tensor, prev_rank)
        return tuple(shape_tensor.tolist())

    return None
```

#### Communication Optimization: Batched P2P

**Standard P2P** (one send + one recv per microbatch):
```python
send_forward(output_tensor, next_rank)
input_tensor = recv_forward(prev_rank)
# Two separate NCCL calls
```

**Batched P2P** (`batch_p2p_comm=True`):
```python
# megatron/core/pipeline_parallel/p2p_communication.py:~200
ops = []

# Batch multiple send/recv operations
if send_tensor_shapes:
    ops.append(torch.distributed.P2POp(
        torch.distributed.isend,
        send_tensor,
        next_rank
    ))

if recv_tensor_shapes:
    ops.append(torch.distributed.P2POp(
        torch.distributed.irecv,
        recv_buffer,
        prev_rank
    ))

# Execute all P2P ops in single batched call
reqs = torch.distributed.batch_isend_irecv(ops)
for req in reqs:
    req.wait()

# Benefits:
# - Single NCCL kernel launch (lower latency)
# - Better scheduling opportunities
# - ~10-15% faster P2P communication
```

### Pipeline Bubble Problem

**Naive (GPipe) Schedule**:
```
Time →
GPU 0: [F0][F1][F2][F3]                [B0][B1][B2][B3]
GPU 1:     [F0][F1][F2][F3]        [B0][B1][B2][B3]
GPU 2:         [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 3:             [F0][F1][F2][F3][B0][B1][B2][B3]

Bubble ratio = (p-1) / p where p = pipeline stages
For p=4: 75% bubble time! (Unacceptable)
```

### 1F1B Schedule (One-Forward-One-Backward)

**Solution**: Interleave forward and backward passes to keep GPUs busy.

```
Time →
GPU 0: [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 1:     [F0][F1][F2][B0][F3][B1][B2][B3]
GPU 2:         [F0][F1][B0][F2][B1][F3][B2][B3]
GPU 3:             [F0][B0][F1][B1][F2][B2][F3][B3]

Bubble ratio = (p-1) / (m+p-1) where m = num_microbatches
For p=4, m=32: 3/35 ≈ 8.6% bubble time (Much better!)
```

**Key insight**: After a short warmup phase, each GPU alternates between forward and backward passes, keeping the pipeline full.

**Memory benefit**: Only keeps `(pipeline_parallel_size - pipeline_parallel_rank - 1) + 1` microbatch activations in memory instead of all `num_microbatches`.

### Virtual Pipeline Parallelism (Interleaved)

**Further optimization**: Each GPU handles **multiple non-contiguous model chunks** (virtual stages).

```
Example: PP=4, Virtual PP=2 (2 virtual stages per GPU)

GPU 0: Layers [0-7, 64-71]
GPU 1: Layers [8-15, 72-79]
GPU 2: Layers [16-23, 80-87]
GPU 3: Layers [24-31, 88-95]

Schedule:
GPU 0: [F0_v0][F1_v0][F0_v1][F1_v1][B0_v0][B1_v0][B0_v1][B1_v1]...
```

**Benefits**:
- **Reduces bubble** by factor of ~virtual_pipeline_size
- Better load balancing across pipeline stages
- For p=4, v=2, m=32: bubble ≈ 4.3% (half of non-interleaved!)

**Trade-off**: Requires more memory to hold multiple model chunks.

**Configuration**:
```bash
--pipeline-model-parallel-size 4
--virtual-pipeline-model-parallel-size 2
```

### Microbatch Configuration

**Critical for minimizing bubbles**: Number of microbatches should be **much larger** than pipeline size.

**Recommendation**:
```python
num_microbatches >= 4 * pipeline_parallel_size
num_microbatches % pipeline_parallel_size == 0  # Avoids uneven distribution
```

**Example**:
```bash
--pipeline-model-parallel-size 8
--num-microbatches 64  # 8× pipeline size
--micro-batch-size 2
--global-batch-size 1024  # = micro_batch_size × num_microbatches × DP_size
```

### P2P Communication Optimization

Two modes for P2P communication:

**1. Overlapped P2P** (`overlap_p2p_comm=True`):
```python
# Overlap send/recv with computation
send_handle = torch.distributed.isend(tensor, dst, async_op=True)
# ... do computation ...
send_handle.wait()
```

**2. Batched P2P** (`batch_p2p_comm=True`):
```python
# Coalesce multiple send/recv into single batched operation
ops = [
    torch.distributed.P2POp(torch.distributed.isend, tensor1, peer1),
    torch.distributed.P2POp(torch.distributed.irecv, tensor2, peer2),
]
reqs = torch.distributed.batch_isend_irecv(ops)
```

**Note**: Cannot use both simultaneously.

**From `schedules.py:901-902`**:
```python
if config.overlap_p2p_comm and config.batch_p2p_comm:
    raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
```

### UCC Backend for Zero-SM Communication

For exposed pipeline parallelism scenarios, **UCC (Unified Collective Communication) backend** offers:

```python
# From schedules.py:976-988
# UCC backend benefits:
# - Zero SM (streaming multiprocessor) resource usage
# - No GPU SM interference with computation
# - Better InfiniBand bandwidth utilization
```

**When to use**: Pipeline communication is on critical path and SM contention is observed.

**Configuration**:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Normally required
# But with UCC: CUDA_DEVICE_MAX_CONNECTIONS > 1 is acceptable
```

### PP Configuration Guidelines

**When to use PP**:
- Very large models (100B+ parameters)
- Inter-node parallelism (PP can tolerate higher latency)
- Memory per GPU is limited

**Constraints**:
- `PP must divide num_layers`
- Requires microbatch count >> pipeline stages
- First and last stages handle data loading/loss computation

**Recommended configurations**:
```bash
# Small model (7B-13B): No PP
--pipeline-model-parallel-size 1

# Medium model (70B) on 64 GPUs:
--pipeline-model-parallel-size 4
--virtual-pipeline-model-parallel-size 2

# Large model (405B) on 512 GPUs:
--pipeline-model-parallel-size 16
--virtual-pipeline-model-parallel-size 4
```

---

## Data Parallelism (DP)

### Concept

**Data Parallelism** is the simplest form: **replicate the entire model** on each GPU and **split the training data** across GPUs. After each backward pass, **gradients are averaged** across all replicas.

**Key operation**: All-reduce of gradients after backward pass.

### Standard DDP (Distributed Data Parallel)

```python
# Each rank computes gradients on its data shard
loss = model(local_batch)
loss.backward()

# All-reduce to average gradients across all DP ranks
for param in model.parameters():
    torch.distributed.all_reduce(
        param.grad,
        op=torch.distributed.ReduceOp.AVG,
        group=data_parallel_group
    )

# Update parameters with averaged gradients
optimizer.step()
```

**Memory requirement**: Each GPU holds full model + full optimizer state.

### ZeRO-Style Sharding (Distributed Optimizer)

**Problem**: With standard DDP, **optimizer states** (momentum, variance for Adam) are replicated across all DP ranks, consuming significant memory.

**Solution**: **Shard optimizer states** across DP ranks (ZeRO optimization).

**From `optimizer/distrib_optimizer.py:1-3264`**, Megatron implements:

**ZeRO-1**: Shard optimizer states only
- Each DP rank owns `1/DP_size` of optimizer states
- After backward: **reduce-scatter** gradients (each rank gets its shard)
- Before forward: **all-gather** parameters from shards

**Benefits**:
- **Optimizer memory**: Reduced by DP_size×
- **Gradient memory**: Can be reduced (with gradient sharding)
- **Parameter memory**: Can be reduced (with parameter sharding)

**Configuration**:
```bash
--use-distributed-optimizer  # Enable ZeRO-style sharding
```

### FSDP (Fully Sharded Data Parallel)

Megatron provides **custom FSDP implementation** that is **~15% faster** than PyTorch native FSDP.

**Sharding strategies** (from `distributed_data_parallel_config.py`):
```python
# No sharding (standard DDP)
--data-parallel-sharding-strategy no_shard

# ZeRO-1: Shard optimizer states only
--data-parallel-sharding-strategy optim

# ZeRO-2: Shard optimizer + gradients
--data-parallel-sharding-strategy optim_grads

# ZeRO-3: Shard optimizer + gradients + parameters
--data-parallel-sharding-strategy optim_grads_params
```

**When to use each level**:
- **no_shard**: Small DP size (≤4), memory not constrained
- **optim**: Medium DP size (4-16), save optimizer memory
- **optim_grads**: Large DP size (16+), save optimizer + gradient memory
- **optim_grads_params**: Extreme scale or memory constrained

**Custom FSDP benefits**:
- 15% faster than PyTorch FSDP
- Better integration with Megatron's other parallelism dimensions
- Optimized for Megatron's buffer management system

### Implementation Deep Dive

Let's examine how gradients are bucketed, communicated, and how ZeRO sharding works in detail.

#### Gradient Buffer and Bucket Formation

**From `param_and_grad_buffer.py:62-100`**, gradients are organized into contiguous buffers:

```python
# megatron/core/distributed/param_and_grad_buffer.py

class _ParamAndGradBuffer:
    """Manages parameter and gradient buckets for efficient communication."""

    def __init__(self, params, data_parallel_group, ...):
        # Calculate total parameter count
        self.numel_total = sum(p.numel() for p in params)

        # Allocate single contiguous gradient buffer
        self.grad_data = torch.zeros(
            self.numel_total,
            dtype=torch.float32,  # Always FP32 for numerical stability!
            device=torch.cuda.current_device(),
            requires_grad=False
        )

        # Create buckets
        self.buckets = []
        current_bucket_params = []
        current_bucket_size = 0

        # Bucket size calculation
        bucket_size = max(40_000_000, 1_000_000 * data_parallel_size)
        # Scales with DP: larger DP → larger buckets to amortize latency

        for param in params:
            current_bucket_params.append(param)
            current_bucket_size += param.numel()

            if current_bucket_size >= bucket_size:
                # Create new bucket
                self.buckets.append(_ParamAndGradBucket(
                    params=current_bucket_params,
                    grad_data=self.grad_data[offset:offset+size],
                    ...
                ))
                # Reset for next bucket
                current_bucket_params = []
                current_bucket_size = 0
```

**Bucket Memory Layout**:
```
Gradient Buffer (single contiguous FP32 tensor):
├─ Bucket 0: [0:40M]         (40M parameters)
├─ Bucket 1: [40M:80M]       (40M parameters)
├─ Bucket 2: [80M:100M]      (20M parameters, last bucket)
└─ Padding: [100M:100.5M]    (aligned to 256 bytes)

Each param.grad points into this buffer (zero-copy view)
```

#### Backward Hook Registration

**How gradients trigger communication** (`distributed_data_parallel.py:441-467`):

```python
# megatron/core/distributed/distributed_data_parallel.py:441-467

def register_grad_ready_hooks(self):
    """Register hooks to detect when each param's gradient is ready."""

    for bucket_idx, bucket in enumerate(self.buffers[dtype].buckets):
        for param_idx, param in enumerate(bucket.params_list):
            # Create closure capturing param and indices
            def make_hook(param, param_idx, bucket):
                def grad_hook(*unused):
                    # Mark this param's gradient as computed
                    bucket.params_with_grad.add(param)

                    # Check if entire bucket is ready
                    if len(bucket.params_with_grad) == len(bucket.params_list):
                        # All grads in bucket ready → trigger communication!
                        self._reduce_scatter_bucket_async(bucket)

                return grad_hook

            # Register post-accumulate hook (fires after .backward())
            param.register_post_accumulate_grad_hook(
                make_hook(param, param_idx, bucket)
            )
```

**Timeline of gradient readiness** for a single bucket:

```
Backward pass (layer by layer, bottom to top):

Time →
Layer N:   compute grad_weight, grad_input  → param1.grad ready
                                            → Hook fires!
Layer N-1: compute grad_weight, grad_input  → param2.grad ready
                                            → Hook fires!
...
Layer N-K: compute grad_weight, grad_input  → paramK.grad ready
                                            → Hook fires!
                                            → ALL bucket grads ready!
                                            → Launch async reduce-scatter!

Bucket communication happens BEFORE backward finishes!
```

#### ZeRO-1: Optimizer State Sharding

**Parameter-to-shard mapping** (`distrib_optimizer.py:~500-700`):

```python
# megatron/core/optimizer/distrib_optimizer.py

class DistributedOptimizer:
    """ZeRO-1 style optimizer state sharding."""

    def __init__(self, optimizer, ...):
        self.data_parallel_group = get_data_parallel_group()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_size = parallel_state.get_data_parallel_world_size()

        # Calculate this rank's shard of parameters
        # Total params: 7B, DP=8 → each rank owns ~875M params
        params_per_rank = total_params // self.dp_size

        # Build parameter-to-shard mapping
        self.param_range_map = {}
        current_offset = 0

        for param in all_params:
            param_size = param.numel()
            # Determine which rank owns this parameter's optimizer state
            owner_rank = current_offset // params_per_rank

            if owner_rank == self.dp_rank:
                # This rank owns this param's optimizer state
                local_offset = current_offset % params_per_rank
                self.param_range_map[param] = (local_offset, param_size)

            current_offset += param_size
```

**Shard distribution example** (7B model, DP=4):

```
Total parameters: 7B = 7,000,000,000

Rank 0 owns optimizer states for: params [0B : 1.75B]
Rank 1 owns optimizer states for: params [1.75B : 3.5B]
Rank 2 owns optimizer states for: params [3.5B : 5.25B]
Rank 3 owns optimizer states for: params [5.25B : 7B]

Each rank stores:
  - Model params: 7B × 2 bytes (FP16) = 14 GB  (replicated)
  - Gradients: 7B × 4 bytes (FP32) = 28 GB     (replicated)
  - Optimizer states (Adam):
      - Momentum: 1.75B × 4 bytes = 7 GB        (sharded!)
      - Variance: 1.75B × 4 bytes = 7 GB        (sharded!)
      - Master weights: 1.75B × 4 bytes = 7 GB  (sharded!)

Total: 14 + 28 + 21 = 63 GB per GPU
vs without sharding: 14 + 28 + 84 = 126 GB per GPU
→ 2× memory savings!
```

#### Reduce-Scatter for ZeRO-1

**From `param_and_grad_buffer.py:~350-420`**:

```python
def reduce_scatter_bucket_async(self, bucket):
    """Reduce-scatter gradients for this bucket."""

    bucket_size = bucket.grad_data.numel()  # e.g., 40M
    shard_size = bucket_size // self.dp_size  # e.g., 10M per rank (DP=4)

    # Allocate output buffer for this rank's shard
    local_shard = torch.empty(
        shard_size,
        dtype=torch.float32,
        device=torch.cuda.current_device()
    )

    # Launch async reduce-scatter
    handle = dist_reduce_scatter_func(
        local_shard,                  # Output: [10M] local shard
        bucket.grad_data,             # Input: [40M] full gradients
        group=self.data_parallel_group,
        async_op=True                 # Non-blocking!
    )

    # Save handle for later synchronization
    self.bucket_comm_handles[bucket.bucket_id] = handle

# Reduce-scatter visual:
# Each rank has full bucket gradients: [40M]
# After reduce-scatter:
#   Rank 0 receives: sum(grad[0:10M]) from all ranks
#   Rank 1 receives: sum(grad[10M:20M]) from all ranks
#   Rank 2 receives: sum(grad[20M:30M]) from all ranks
#   Rank 3 receives: sum(grad[30M:40M]) from all ranks
```

#### FSDP Memory Savings Example

**Memory breakdown** (7B model, DP=4, FP16 params):

```
Standard DDP:
  - Params: 7B × 2 bytes = 14 GB     (full model)
  - Grads: 7B × 4 bytes = 28 GB      (FP32)
  - Optimizer: 7B × 12 bytes = 84 GB (Adam FP32)
  Total: 126 GB per GPU

ZeRO-1 (optim only):
  - Params: 14 GB (full)
  - Grads: 28 GB (full)
  - Optimizer: 21 GB (sharded 1/4)
  Total: 63 GB per GPU  (2× savings)

ZeRO-2 (optim + grads):
  - Params: 14 GB (full)
  - Grads: 7 GB (sharded 1/4)
  - Optimizer: 21 GB (sharded 1/4)
  Total: 42 GB per GPU  (3× savings)

ZeRO-3 (optim + grads + params):
  - Params: 3.5 GB (sharded 1/4)
  - Grads: 7 GB (sharded 1/4)
  - Optimizer: 21 GB (sharded 1/4)
  Total: 31.5 GB per GPU  (4× savings!)

Trade-off: More communication (all-gather params every layer)
```

### Gradient Bucketing

**From `param_and_grad_buffer.py:62-100`**, gradients are grouped into **buckets** for efficient communication:

```python
# Default bucket size calculation
bucket_size = max(40_000_000, 1_000_000 * data_parallel_size)
# 40M parameters minimum, scaled by DP size

# Why bucketing?
# 1. Amortize communication overhead (one collective per bucket vs per parameter)
# 2. Enable asynchronous communication (bucket ready → launch async reduce)
# 3. Better alignment for NCCL performance
```

**Bucket alignment requirements**:
- **256-byte alignment** (128 values at FP16) for cuBLAS algorithm selection
- **2^16 alignment** for NCCL high bandwidth at large scale
- **DP-size divisibility** for distributed optimizer sharding

### Gradient Reduction Overlap

**Critical optimization**: Don't wait for entire backward pass to complete before starting gradient reduction.

**Strategy** (from `distributed_data_parallel.py:441-467`):
```python
# Register backward hook on each parameter
def grad_ready_hook(param):
    # When this param's gradient is computed:
    bucket = find_bucket_for_param(param)
    bucket.mark_grad_ready(param)

    if bucket.all_grads_ready():
        # Launch async reduce-scatter immediately!
        bucket.async_reduce_scatter()

# Hook is triggered during backward pass
# Gradients are reduced while backward continues on other layers
```

**Result**: Gradient communication **overlapped with backward computation**, hiding 80-90% of communication latency. See [Communication Overlap](02-communication-overlap.md#gradient-reduction-overlap) for details.

### DP Configuration Guidelines

**DP size calculation**:
```python
DP_size = Total_GPUs / (TP_size × PP_size × CP_size × EP_size)
```

**When to use DP**:
- Increase overall throughput (more data processed per step)
- Fill remaining GPUs after TP/PP are configured
- DP provides easy scaling with good efficiency

**Recommended configurations**:
```bash
# Small DP (1-8): No sharding needed
--data-parallel-sharding-strategy no_shard

# Medium DP (8-64): Shard optimizer
--use-distributed-optimizer
--data-parallel-sharding-strategy optim
--overlap-grad-reduce  # Enable overlap

# Large DP (64+): Full FSDP
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--overlap-grad-reduce
--overlap-param-gather
```

---

## Context Parallelism (CP)

### Concept

**Context Parallelism** addresses the challenge of **long sequence lengths** by splitting the sequence dimension across GPUs.

**Problem**: Even with TP and PP, a single sequence's activations may not fit in GPU memory. For example:
- Sequence length: 32K tokens
- Batch size: 8
- Hidden size: 4096
- Attention activations: O(seq_len²) memory

**Solution**: Partition sequences across multiple GPUs and use **ring-based communication** for attention computation.

### How It Works

```
Original sequence: [token_0, token_1, ..., token_31999]  # 32K tokens

With CP=4:
GPU 0: [token_0, ..., token_7999]     # First 8K
GPU 1: [token_8000, ..., token_15999] # Second 8K
GPU 2: [token_16000, ..., token_23999] # Third 8K
GPU 3: [token_24000, ..., token_31999] # Fourth 8K
```

### Ring Attention

For self-attention, each GPU needs to attend to **all tokens** in the sequence, not just its local partition.

**Implementation**: Ring-based all-to-all communication:
```
Step 1: GPU 0 computes attention with local K/V
Step 2: Rotate K/V: GPU 0 sends to GPU 1, GPU 3 sends to GPU 0
Step 3: GPU 0 computes attention with received K/V
Step 4: Continue rotation until all K/V pairs processed
```

**Result**: Each GPU computes attention over the full sequence length while only storing `seq_len / CP_size` activations.

### Hierarchical Context Parallelism

For even longer sequences, **hierarchical CP** uses multiple levels:

```bash
--context-parallel-size 8
--hierarchical-context-parallel-sizes 2 4
# Level 1: 2-way CP within node
# Level 2: 4-way CP across nodes
```

**Communication pattern**:
- **Intra-node**: All-to-all over NVLink (low latency)
- **Inter-node**: All-to-all over InfiniBand (higher latency but batched)

### CP Communication Types

From configuration options:
```bash
--cp-comm-type p2p         # Point-to-point ring communication
--cp-comm-type a2a         # All-to-all collective
--cp-comm-type allgather   # All-gather + local computation
```

**When to use each**:
- **p2p**: Best for very long sequences, minimizes memory
- **a2a**: Better for moderate sequences, potentially faster
- **allgather**: Simplest, but requires more memory

### CP Configuration Guidelines

**When to use CP**:
- Sequence length > 8K tokens
- Memory-bound on activations even with TP/PP
- Training with long context (32K, 64K, 128K tokens)

**Constraints**:
- `CP must divide sequence_length`
- Adds communication overhead (ring passes)
- Works best with FlashAttention for memory-efficient attention

**Recommended configurations**:
```bash
# 8K-16K sequences:
--context-parallel-size 2
--cp-comm-type a2a

# 32K-64K sequences:
--context-parallel-size 4
--hierarchical-context-parallel-sizes 2 2
--cp-comm-type p2p

# 128K+ sequences:
--context-parallel-size 8
--hierarchical-context-parallel-sizes 2 4
```

### Implementation Deep Dive

#### CP Integration with Attention

Context Parallelism is implemented **exclusively through Transformer Engine (TE)**, not in Megatron's native attention.

**From `megatron/core/transformer/dot_product_attention.py:58-59`**:
```python
assert (
    self.config.context_parallel_size == 1
), "Context parallelism is only supported by TEDotProductAttention!"
```

**Why TE only?** CP requires sophisticated ring-based or all-to-all communication during attention computation, which TE implements with optimized CUDA kernels and communication primitives.

#### TEDotProductAttention Setup

**From `megatron/core/extensions/transformer_engine.py:852-937`**:

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str = "p2p",  # Default: point-to-point ring
        pg_collection: ProcessGroupCollection = None,
    ):
        # CP process groups setup
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp', 'hcp']
            )

        # Pass CP configuration to TE
        extra_kwargs = {
            "cp_group": pg_collection.cp,  # CP process group
            "cp_global_ranks": torch.distributed.get_process_group_ranks(
                pg_collection.cp
            ),
            "cp_stream": TEDotProductAttention.cp_stream,  # Dedicated CUDA stream
        }

        # Set communication type
        if cp_comm_type == "a2a+p2p":
            # Hierarchical CP: all-to-all + point-to-point
            extra_kwargs["cp_comm_type"] = "a2a+p2p"
            extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups()
        else:
            # Simple CP: p2p, a2a, or allgather
            extra_kwargs["cp_comm_type"] = cp_comm_type
```

**Key Points**:
1. **Dedicated CUDA stream** (`cp_stream`) for CP communication to enable overlap
2. **Process group configuration** passed to TE for distributed operations
3. **Communication type selection** determines trade-offs

#### Ring Attention Implementation (cp_comm_type="p2p")

Ring attention enables **memory-efficient long-context** by streaming K/V chunks across GPUs.

**Conceptual flow** (TE internal implementation):

```python
# Example: CP=4, seq_len=32K, local_seq_len=8K per GPU

# Step 1: Local attention with own K/V chunk
# GPU 0: Q[0:8K] attends to K[0:8K], V[0:8K]
# GPU 1: Q[8K:16K] attends to K[8K:16K], V[8K:16K]
# GPU 2: Q[16K:24K] attends to K[16K:24K], V[16K:24K]
# GPU 3: Q[24K:32K] attends to K[24K:32K], V[24K:32K]

output = torch.zeros_like(Q_local)
max_score = torch.full((batch, heads, local_seq_len), -inf)
sum_exp = torch.zeros((batch, heads, local_seq_len))

# Initial local computation
attn_scores = Q_local @ K_local.transpose(-2, -1) / sqrt(d_k)
attn_probs = softmax(attn_scores, dim=-1)
output += attn_probs @ V_local

# Step 2-4: Ring rotations (CP-1 steps)
for ring_step in range(1, CP_size):
    # P2P communication: rotate K/V tensors
    # GPU i sends to GPU (i+1) % CP_size
    # GPU i receives from GPU (i-1) % CP_size
    K_remote, V_remote = ring_exchange_p2p(K_local, V_local, cp_group)

    # Compute attention with remote K/V
    attn_scores_remote = Q_local @ K_remote.transpose(-2, -1) / sqrt(d_k)

    # Online softmax: update running max and sum
    new_max = torch.max(max_score, attn_scores_remote.max(dim=-1, keepdim=True))
    exp_scores = torch.exp(attn_scores_remote - new_max)

    # Reweight previous output with new max
    correction = torch.exp(max_score - new_max)
    output = output * correction + (exp_scores @ V_remote)
    sum_exp = sum_exp * correction + exp_scores.sum(dim=-1, keepdim=True)
    max_score = new_max

    # Prepare for next rotation
    K_local, V_local = K_remote, V_remote

# Final normalization
output = output / sum_exp

# Result: Each GPU has attended to ALL 32K tokens
# Memory: Only 8K local + 8K remote K/V at any time (vs 32K full sequence)
```

**Memory savings**:
```
Without CP: seq_len² attention matrix
  32K tokens: 32K × 32K = 1B elements × 2 bytes (FP16) = 2 GB per batch

With CP=4: (seq_len/CP)² + seq_len × (seq_len/CP) local + remote
  Local: 8K × 8K = 64M elements
  Remote (streaming): 8K × 8K = 64M elements
  Total: 128M elements × 2 bytes = 256 MB per batch (8× savings!)
```

#### All-to-All Implementation (cp_comm_type="a2a")

All-to-all exchanges **all K/V chunks simultaneously**, trading memory for speed.

**Communication pattern**:
```python
# Before all-to-all (each GPU has local K/V)
# GPU 0: K[0:8K], V[0:8K]
# GPU 1: K[8K:16K], V[8K:16K]
# GPU 2: K[16K:24K], V[16K:24K]
# GPU 3: K[24K:32K], V[24K:32K]

# After all-to-all (each GPU has ALL K/V)
K_full = torch.distributed.all_to_all(K_local, cp_group)
V_full = torch.distributed.all_to_all(V_local, cp_group)
# GPU 0-3: K[0:32K], V[0:32K] (full sequence!)

# Standard attention computation
attn_scores = Q_local @ K_full.transpose(-2, -1) / sqrt(d_k)
attn_probs = softmax(attn_scores, dim=-1)
output = attn_probs @ V_full
```

**Trade-offs**:
- **Faster**: One all-to-all vs CP-1 ring steps
- **More memory**: Must hold full K/V (seq_len size) instead of streaming
- **Better for moderate sequences**: 8K-16K where memory is not critical

#### Hierarchical CP (cp_comm_type="a2a+p2p")

Combines **intra-node all-to-all** (fast NVLink) with **inter-node p2p** (slower IB).

**Example**: CP=8 with `--hierarchical-context-parallel-sizes 2 4`

```python
# Level 1 (intra-node): CP=2 per node, 4 nodes
# Level 2 (inter-node): CP=4 across nodes

# Within each node: Use all-to-all over NVLink
#   Node 0: GPU 0-1 exchange K/V chunks (fast!)
#   Node 1: GPU 2-3 exchange K/V chunks
#   Node 2: GPU 4-5 exchange K/V chunks
#   Node 3: GPU 6-7 exchange K/V chunks

# Across nodes: Use ring p2p over InfiniBand
#   GPU 0,1 (node 0) ↔ GPU 2,3 (node 1) ↔ GPU 4,5 (node 2) ↔ GPU 6,7 (node 3)
#   Ring: 3 steps of inter-node communication (vs 7 steps flat ring!)

# Total communication:
#   1 intra-node all-to-all (NVLink, ~5 μs latency)
#   3 inter-node p2p ring steps (IB, ~10-20 μs each)
# vs flat CP=8 ring:
#   7 ring steps (potentially all inter-node)
```

**Benefits**:
- **Exploits topology**: Fast intra-node, minimize inter-node
- **Reduced latency**: Fewer inter-node hops
- **Scalability**: Supports 128K+ sequences across 8+ nodes

#### CP Communication Overlap

**Dedicated CUDA stream** enables overlap with computation:

```python
# From transformer_engine.py:840, 918
class TEDotProductAttention:
    cp_stream: torch.cuda.Stream = None  # Class-level shared stream

    def __init__(self, ...):
        if TEDotProductAttention.cp_stream is None:
            TEDotProductAttention.cp_stream = torch.cuda.Stream()
        extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
```

**Overlap strategy** (within TE):
```python
# Main compute stream
with torch.cuda.stream(main_stream):
    Q_local = project_q(hidden_states)  # Compute Q projection
    K_local = project_k(hidden_states)  # Compute K projection
    V_local = project_v(hidden_states)  # Compute V projection

# CP communication stream (overlapped!)
with torch.cuda.stream(cp_stream):
    K_remote, V_remote = ring_exchange_p2p(K_local, V_local)

# Synchronize before attention computation
main_stream.wait_stream(cp_stream)

# Attention computation uses both local and remote K/V
output = compute_attention(Q_local, K_remote, V_remote)
```

**Result**: QKV projection computation **overlaps** with K/V rotation, hiding ~50% of communication latency.

#### Tensor Shapes Through CP

**Example**: Sequence length 32K, CP=4, batch=4, hidden=4096, heads=32

```python
# Input to attention layer
hidden_states: [8K, 4, 4096]  # [local_seq, batch, hidden]
# Local sequence length: 32K / 4 = 8K

# After QKV projection
Q: [8K, 4, 32, 128]  # [local_seq, batch, heads, head_dim]
K: [8K, 4, 32, 128]
V: [8K, 4, 32, 128]

# During ring attention (step i)
K_remote: [8K, 4, 32, 128]  # Received from neighbor
V_remote: [8K, 4, 32, 128]

# Attention scores
attn_scores: [4, 32, 8K, 8K]  # [batch, heads, local_q, local_k]
# Computed incrementally for each ring step

# Output
output: [8K, 4, 4096]  # [local_seq, batch, hidden]
# Each GPU outputs its local sequence partition
# Aggregated result represents attention over full 32K sequence
```

#### CP Configuration Summary

| **Communication Type** | **Memory** | **Speed** | **Best For** |
|----------------------|------------|-----------|--------------|
| **p2p** (ring) | Lowest (2/CP) | Slower (CP-1 steps) | Very long sequences (64K+) |
| **a2a** (all-to-all) | Highest (full seq) | Faster (1 step) | Moderate sequences (8K-16K) |
| **a2a+p2p** (hierarchical) | Medium | Medium | Multi-node long sequences (32K-128K) |

**Rule of thumb**:
```python
if seq_len <= 16K and memory_available > 2GB:
    use cp_comm_type="a2a"
elif num_nodes > 4 and seq_len >= 32K:
    use cp_comm_type="a2a+p2p" with hierarchical sizes
else:
    use cp_comm_type="p2p"
```

---

## Expert Parallelism (EP)

### Concept

**Expert Parallelism** is specific to **Mixture of Experts (MoE)** models. In MoE, each layer has multiple "expert" networks, and each token is routed to a subset of experts.

**Problem**: With 8-64 experts per layer, even a single layer's experts may not fit in GPU memory.

**Solution**: Distribute experts across GPUs. Each GPU holds a subset of experts.

### How MoE Works

```
Input tokens: [t0, t1, t2, t3, t4, t5, t6, t7]
Experts: [E0, E1, E2, E3, E4, E5, E6, E7]  # 8 experts

Router decides:
  t0 → E1, E4
  t1 → E2, E7
  t2 → E0, E3
  ...

With EP=4:
  GPU 0: E0, E1
  GPU 1: E2, E3
  GPU 2: E4, E5
  GPU 3: E6, E7
```

### All-to-All Communication

**Key operation**: **All-to-all** to route tokens to the correct expert GPU.

```python
# Forward pass
# 1. All-to-all: Send tokens to GPUs that have their assigned experts
token_to_expert = all_to_all_send_tokens(tokens, expert_assignments)

# 2. Each GPU processes its experts
expert_outputs = []
for expert in local_experts:
    expert_outputs.append(expert(token_to_expert[expert]))

# 3. All-to-all: Return processed tokens to original GPUs
final_outputs = all_to_all_return_tokens(expert_outputs)
```

**Communication pattern**: **Irregular** (different number of tokens per expert), requires careful load balancing.

### Load Balancing Challenge

**Problem**: If routing is imbalanced, some expert GPUs process many tokens while others are idle.

Example of **bad** routing:
```
GPU 0 (E0, E1): 500 tokens  ← Overloaded!
GPU 1 (E2, E3): 100 tokens
GPU 2 (E4, E5): 150 tokens
GPU 3 (E6, E7): 50 tokens   ← Idle most of the time!
```

**Solution**: **Aux-loss-free load balancing** (used in DeepSeek-V3)
- Router is trained to balance load without explicit aux loss
- Ensures approximately equal tokens per expert
- Critical for GPU utilization in MoE models

### Grouped GEMM Optimization

**Challenge**: Each expert processes a different number of tokens → variable-sized matrix multiplications.

**Solution**: **Grouped GEMM** batches all expert GEMMs into a single efficient operation.

From `moe/experts.py:54`:
```python
# Instead of:
for expert in experts:
    output = expert.linear(input[expert])  # Inefficient!

# Use grouped GEMM:
output = grouped_gemm(
    inputs=[input[e] for e in experts],
    weights=[expert.weight for expert in experts]
)
# 3-5× faster!
```

**Library**: `nv-grouped-gemm` (CUTLASS-based, GPU-optimized)

### EP + TP Interaction

**CRITICAL**: When combining EP with TP, **Sequence Parallelism MUST be enabled**.

**Reason**: EP's all-to-all requires full sequence, but TP alone keeps sequence partitioned.

**Configuration**:
```bash
--expert-model-parallel-size 8
--tensor-model-parallel-size 2
--sequence-parallel  # Required!
```

### EP Configuration Guidelines

**When to use EP**:
- Training MoE models (Mixtral, DeepSeek-V3, Qwen-MoE)
- Number of experts > 8
- Experts don't fit in single GPU memory

**Constraints**:
- `EP must divide num_experts`
- Requires sequence parallelism if TP > 1
- Load balancing is critical for efficiency

**Recommended configurations**:
```bash
# Mixtral 8×7B (8 experts):
--num-experts 8
--expert-model-parallel-size 4
--moe-grouped-gemm  # Enable grouped GEMM

# DeepSeek-V3 671B (256 experts):
--num-experts 256
--expert-model-parallel-size 64
--tensor-model-parallel-size 2
--sequence-parallel
--moe-aux-loss-free-balancing  # Critical!
```

### Implementation Deep Dive

Expert Parallelism (EP) is the most complex parallelism dimension, involving **token routing**, **all-to-all communication**, and **grouped GEMM** for efficient expert computation.

#### MoE Token Dispatcher Architecture

The **Token Dispatcher** handles the 3-phase token routing process.

**From `megatron/core/transformer/moe/token_dispatcher.py:36-44`**, notation:
```python
# Notation used throughout MoE implementation:
# H: hidden size
# B: micro batch size
# S: sequence length
# TP: tensor model parallel size
# EP: expert model parallel size
# num_local_tokens: S/TP*B (tokens on this GPU before EP)
# num_global_tokens: num_local_tokens*TP*EP (total tokens across EP group)
```

**Three-phase dispatch**:
```python
class MoETokenDispatcher:
    @abstractmethod
    def dispatch_preprocess(self, tokens, routing_map, probs):
        """Phase 1: Local preprocessing (no communication)"""

    @abstractmethod
    def token_dispatch(self, hidden_states, probs):
        """Phase 2: All-to-all communication"""

    @abstractmethod
    def dispatch_postprocess(self, hidden_states, probs):
        """Phase 3: Local postprocessing (no communication)"""
```

**Why 3 phases?** Enables **communication-computation overlap** (see [Communication Overlap](02-communication-overlap.md#moe-overlap)).

#### Router and Expert Assignment

**From `megatron/core/transformer/moe/router.py:77-99`**, router gating:

```python
class TopKRouter(Router):
    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate."""
        # Router weights: [num_experts, hidden_size]
        # Input: [S*B/TP, H]

        # Convert to routing dtype (FP32 for numerical stability)
        router_dtype = torch.float32 if config.moe_router_dtype == 'fp32' else input.dtype

        # Compute logits: [S*B/TP, num_experts]
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    def routing(self, logits):
        """Top-K expert selection."""
        # logits: [S*B/TP, num_experts]
        k = config.moe_router_topk  # Typically 2

        # Select top-k experts per token
        scores, indices = torch.topk(logits, k, dim=-1)
        # scores: [S*B/TP, k]
        # indices: [S*B/TP, k] (which experts)

        # Normalize scores (softmax over top-k)
        probs = F.softmax(scores, dim=-1)

        # Create routing map (sparse binary matrix)
        routing_map = torch.zeros_like(logits)  # [S*B/TP, num_experts]
        routing_map.scatter_(1, indices, 1.0)   # Set selected experts to 1

        return probs, routing_map
```

**Example**: 4 tokens, 8 experts, top-2 routing:
```python
# Input tokens: [t0, t1, t2, t3]
# Router decides:
#   t0 → experts [1, 4] with probs [0.6, 0.4]
#   t1 → experts [2, 7] with probs [0.55, 0.45]
#   t2 → experts [0, 3] with probs [0.7, 0.3]
#   t3 → experts [1, 5] with probs [0.65, 0.35]

# routing_map: [4, 8] sparse matrix
# [[0, 1, 0, 0, 1, 0, 0, 0],   # t0
#  [0, 0, 1, 0, 0, 0, 0, 1],   # t1
#  [1, 0, 0, 1, 0, 0, 0, 0],   # t2
#  [0, 1, 0, 0, 0, 1, 0, 0]]   # t3
```

#### All-to-All Token Dispatch

**From `megatron/core/transformer/moe/token_dispatcher.py:607-628`**, all-to-all communication:

```python
class AllToAllTokenDispatcher(MoETokenDispatcher):
    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        """Perform all-to-all communication for dispatching tokens."""

        # Synchronize metadata (tokens per expert) to all EP ranks
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )

        # All-to-all: send tokens to GPUs with their assigned experts
        global_input_tokens = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,  # How many tokens to send to each EP rank
            self.input_splits,   # How many tokens to receive from each EP rank
        )

        return global_input_tokens, permuted_probs
```

**Detailed example**: EP=4, 8 experts (2 per GPU), top-2 routing

```python
# Setup
EP_size = 4
num_experts = 8
experts_per_gpu = 2
# GPU 0: Experts [0, 1]
# GPU 1: Experts [2, 3]
# GPU 2: Experts [4, 5]
# GPU 3: Experts [6, 7]

# Input: 8 tokens on GPU 0
tokens = [t0, t1, t2, t3, t4, t5, t6, t7]  # [8, hidden_size]

# Router assigns:
# t0 → [E1, E4] → [GPU0, GPU2]
# t1 → [E2, E7] → [GPU1, GPU3]
# t2 → [E0, E3] → [GPU0, GPU1]
# t3 → [E5, E6] → [GPU2, GPU3]
# ... (more tokens)

# Step 1: Permutation - group tokens by destination expert
permuted_tokens = [
    # For GPU 0 (E0, E1):
    t2_for_E0, t0_for_E1,
    # For GPU 1 (E2, E3):
    t1_for_E2, t2_for_E3,
    # For GPU 2 (E4, E5):
    t0_for_E4, t3_for_E5,
    # For GPU 3 (E6, E7):
    t3_for_E6, t1_for_E7,
]

# Step 2: All-to-all split calculation
output_splits = [2, 2, 2, 2]  # Send 2 tokens to each GPU
input_splits = [?, ?, ?, ?]   # Receive (determined by router on other GPUs)

# Step 3: All-to-all communication
# GPU 0 sends: 2 tokens to GPU 0 (local), 2 to GPU 1, 2 to GPU 2, 2 to GPU 3
# GPU 0 receives: tokens from all 4 GPUs destined for E0, E1
global_tokens = all_to_all(
    ep_group,
    permuted_tokens,
    output_splits=[2, 2, 2, 2],
    input_splits=[3, 2, 2, 1],  # Example: received 8 total tokens for E0, E1
)

# Result on GPU 0: [8 tokens, hidden_size] to be processed by E0 and E1
```

**Communication pattern**:
```
Before all-to-all:
  GPU 0: [t2(E0), t0(E1), t1(E2), t2(E3), t0(E4), t3(E5), t3(E6), t1(E7)]
  GPU 1: [... similar distribution of its local tokens ...]
  GPU 2: [... similar distribution ...]
  GPU 3: [... similar distribution ...]

After all-to-all:
  GPU 0: [all tokens assigned to E0 or E1 from all GPUs]
  GPU 1: [all tokens assigned to E2 or E3 from all GPUs]
  GPU 2: [all tokens assigned to E4 or E5 from all GPUs]
  GPU 3: [all tokens assigned to E6 or E7 from all GPUs]
```

#### Grouped GEMM for Expert Computation

**Challenge**: Each expert processes a **different number of tokens** → variable-sized matrix multiplications.

**Naïve approach** (slow!):
```python
# Process each expert separately
expert_outputs = []
for i, expert in enumerate(local_experts):
    tokens_for_expert = tokens[token_map == i]  # Variable size!
    output = expert(tokens_for_expert)          # Different GEMM sizes
    expert_outputs.append(output)
# Problem: Each GEMM is small, GPU underutilized!
```

**Grouped GEMM approach** (fast!):

**From `megatron/core/transformer/moe/experts.py:100-150`**:

```python
class GroupedMLP(MegatronModule):
    """Efficient implementation using GroupedGEMM."""

    def __init__(self, num_local_experts, config):
        self.num_local_experts = num_local_experts  # e.g., 2 experts per GPU

        # Initialize weights for all local experts
        # fc1: [num_local_experts, hidden_size, ffn_hidden_size]
        # fc2: [num_local_experts, ffn_hidden_size, hidden_size]
        self.weight1 = Parameter(
            torch.empty((num_local_experts, hidden_size, ffn_hidden_size))
        )
        self.weight2 = Parameter(
            torch.empty((num_local_experts, ffn_hidden_size, hidden_size))
        )

    def forward(self, tokens, tokens_per_expert, probs):
        """
        Args:
            tokens: [total_tokens, hidden_size]
            tokens_per_expert: [num_local_experts] - how many tokens each expert gets
            probs: [total_tokens] - routing probabilities
        """
        # Step 1: Grouped GEMM for fc1
        # Instead of looping over experts, batch all GEMMs!
        fc1_output = grouped_gemm.ops.gmm(
            tokens,                # Input: [total_tokens, hidden_size]
            self.weight1,          # Weights: [num_local_experts, hidden_size, ffn_hidden_size]
            tokens_per_expert,     # Split: how many tokens per expert
            trans_b=False
        )
        # Output: [total_tokens, ffn_hidden_size]

        # Step 2: Activation function
        fc1_output = self.activation_func(fc1_output) * probs.unsqueeze(-1)

        # Step 3: Grouped GEMM for fc2
        output = grouped_gemm.ops.gmm(
            fc1_output,
            self.weight2,
            tokens_per_expert,
            trans_b=False
        )
        # Output: [total_tokens, hidden_size]

        return output
```

**Example**: 2 experts, different token counts

```python
# GPU 2 has experts [E4, E5]
tokens_for_E4 = 120  # E4 gets 120 tokens
tokens_for_E5 = 180  # E5 gets 180 tokens
total_tokens = 300

# Input tokens: [300, 4096]
# weight1: [2, 4096, 16384] (2 experts, hidden=4096, ffn=16384)
# tokens_per_expert: [120, 180]

# Grouped GEMM execution (conceptual):
# 1. Kernel splits input into chunks: [0:120] for E4, [120:300] for E5
# 2. Launches kernel with metadata:
#    - Expert 0: GEMM([120, 4096] × [4096, 16384])
#    - Expert 1: GEMM([180, 4096] × [4096, 16384])
# 3. Executes both GEMMs in parallel on GPU
# 4. Concatenates outputs: [300, 16384]

# Performance: 3-5× faster than sequential expert execution!
```

**From `megatron/core/transformer/moe/grouped_gemm_util.py:1-22`**, library:
```python
try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

# Uses CUTLASS-based optimized kernels
# Install: pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

#### All-to-All Return Communication

After expert computation, tokens must return to their **original GPUs**.

**From token dispatcher** (combine phase):
```python
def token_combine(self, hidden_states):
    """Return expert outputs to original GPUs."""

    # Reverse all-to-all: send tokens back
    output = all_to_all(
        self.ep_group,
        hidden_states,
        self.input_splits,   # Now these are output splits (reversed)
        self.output_splits,  # Now these are input splits (reversed)
    )

    # Unpermute: restore original token order
    output = unpermute(output, self.reversed_permutation_mapping)

    return output
```

**Full EP flow summary**:
```python
# 1. Forward: token dispatch
router_output, routing_map = router(hidden_states)
tokens, probs = dispatcher.dispatch_preprocess(hidden_states, routing_map, router_output)
tokens, probs = dispatcher.token_dispatch(tokens, probs)  # All-to-all send
tokens, probs, tokens_per_expert = dispatcher.dispatch_postprocess(tokens, probs)

# 2. Expert computation
expert_output = grouped_mlp(tokens, tokens_per_expert, probs)

# 3. Backward: token combine
output = dispatcher.combine_preprocess(expert_output)
output = dispatcher.token_combine(output)  # All-to-all return
output = dispatcher.combine_postprocess(output)
```

#### Load Balancing

**Problem**: If routing is imbalanced, some expert GPUs are overloaded while others idle.

**Example of bad routing**:
```python
# EP=4, 8 experts, 1000 tokens
# GPU 0 (E0, E1): 500 tokens ← Overloaded! (62.5% utilization)
# GPU 1 (E2, E3): 100 tokens (12.5%)
# GPU 2 (E4, E5): 150 tokens (18.75%)
# GPU 3 (E6, E7): 250 tokens (31.25%)

# Result: GPU 0 is bottleneck, others wait (25% average utilization!)
```

**DeepSeek-V3 aux-loss-free balancing**:

```python
# Traditional approach: Add auxiliary loss to encourage balance
aux_loss = load_balance_loss(routing_probs, expert_assignments)
total_loss = task_loss + α * aux_loss

# DeepSeek-V3 approach: Train router to balance WITHOUT explicit loss
# Uses bias initialization and normalization techniques
# Result: 95%+ load balance without aux loss overhead
```

**Configuration**:
```bash
--moe-aux-loss-free-balancing  # Enable DeepSeek-V3 balancing
```

#### EP + TP Interaction

**Critical constraint**: When combining EP with TP, **Sequence Parallelism MUST be enabled**.

**Why?** EP's all-to-all requires **full sequence** per GPU, but TP alone keeps sequences partitioned.

**From `megatron/core/transformer/moe/token_dispatcher.py:250-260`**:
```python
if self.tp_size > 1 or self.ep_size > 1:
    # Gather tokens from all TP ranks before EP all-to-all
    self.routing_map = gather_from_sequence_parallel_region(
        self.routing_map,
        group=self.tp_ep_group  # TP × EP process group
    )
    # Input: [S/TP*B, num_experts]
    # Output: [S*B*EP, num_experts]

    # Now all-to-all can distribute full sequences across EP
```

**Configuration requirement**:
```bash
--expert-model-parallel-size 8
--tensor-model-parallel-size 2
--sequence-parallel  # REQUIRED! Enables gather before EP all-to-all
```

#### Tensor Shapes Through EP

**Example**: Mixtral 8×7B, EP=4, TP=1, 1000 tokens, hidden=4096

```python
# 1. Router input
hidden_states: [1000, 4096]  # [num_tokens, hidden]

# 2. Router output
logits: [1000, 8]            # [num_tokens, num_experts]
routing_map: [1000, 8]       # Sparse binary (top-2 per token)
probs: [1000, 2]             # Top-2 probabilities per token

# 3. After dispatch preprocess (permutation)
# Tokens grouped by destination expert
permuted_tokens: [2000, 4096]  # 1000 tokens × 2 experts/token = 2000 token-expert pairs

# 4. All-to-all splits (example)
# GPU 0 sends:  [500, 4096] (tokens for E0, E1 across all GPUs)
# GPU 1 sends:  [500, 4096] (tokens for E2, E3)
# GPU 2 sends:  [500, 4096] (tokens for E4, E5)
# GPU 3 sends:  [500, 4096] (tokens for E6, E7)

# 5. After all-to-all (GPU 0 receives)
global_tokens: [520, 4096]     # Received tokens for E0 and E1
tokens_per_expert: [270, 250]  # E0 gets 270, E1 gets 250

# 6. Grouped GEMM (GPU 0)
# weight1: [2, 4096, 16384]   # 2 local experts
expert_fc1_out: [520, 16384]  # After first layer
expert_output: [520, 4096]    # After second layer

# 7. All-to-all return (send back to original GPUs)
returned_tokens: [500, 4096]  # Back to original GPU 0 tokens

# 8. Unpermute and combine
final_output: [1000, 4096]    # Original token order restored
```

#### EP Performance Optimization Summary

| **Optimization** | **Speedup** | **Where** |
|-----------------|------------|-----------|
| **Grouped GEMM** | 3-5× | Expert computation |
| **All-to-all overlap** | 1.2-1.5× | Communication hiding |
| **Load balancing** | 1.5-2× | Avoid idle GPUs |
| **FP8 experts** | 1.5-2× | Reduce memory/communication |

**Combined result**: 10-15× speedup over naïve sequential expert execution!

---

## Multi-Dimensional Parallelism

### Combining Parallelism Dimensions

The power of Megatron is the ability to **compose** all 5 parallelism dimensions:

```
Total GPUs = TP × PP × CP × EP × DP

Each dimension serves a specific purpose:
- TP: Split large layers (intra-node, low latency)
- PP: Split model depth (inter-node, high throughput)
- CP: Split long sequences (reduce activation memory)
- EP: Split MoE experts (MoE-specific)
- DP: Data throughput (fill remaining GPUs)
```

### Example Configurations

**LLaMA-3 8B on 8 GPUs (single node)**:
```bash
TP=1, PP=1, CP=2, DP=4
# Model fits in single GPU, use CP for long sequences, DP for throughput
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 2
# DP = 8 / (1×1×2) = 4 automatically
```

**LLaMA-3 70B on 64 GPUs**:
```bash
TP=4, PP=4, CP=2, DP=2
# 2D parallelism (TP+PP) for model, CP for sequences
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 4
--virtual-pipeline-model-parallel-size 2
--context-parallel-size 2
# DP = 64 / (4×4×2) = 2
```

**LLaMA-3.1 405B on 1024 GPUs**:
```bash
TP=8, PP=8, CP=2, DP=16
# 3D parallelism for massive scale
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 8
--virtual-pipeline-model-parallel-size 4
--context-parallel-size 2
# DP = 1024 / (8×8×2) = 16
```

**Mixtral 8×7B on 64 GPUs**:
```bash
TP=1, PP=4, EP=8, DP=2
# EP for experts, PP for depth
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 4
--expert-model-parallel-size 8
--num-experts 8
# DP = 64 / (1×4×8) = 2
```

**DeepSeek-V3 671B on 1024 GPUs**:
```bash
TP=2, PP=16, EP=64, DP=0.5 (gradient accumulation)
# Large MoE with heavy EP
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 16
--virtual-pipeline-model-parallel-size 4
--expert-model-parallel-size 64
--num-experts 256
# DP = 1024 / (2×16×64) = 0.5 → Use gradient accumulation
```

### Parallelism Selection Decision Tree

```
1. Does model fit in single GPU? (7B-13B)
   Yes → TP=1, PP=1, use DP for scaling
   No → Continue

2. Does model fit in single node? (13B-70B)
   Yes → Use TP (2-8), DP for remaining GPUs
   No → Continue

3. Is model very large? (70B-405B)
   Yes → Use TP (4-8) + PP (4-16)
   Also consider:
     - Virtual PP to reduce bubbles
     - CP if long sequences
     - DP to fill remaining GPUs

4. Is model MoE?
   Yes → Add EP (divide num_experts)
   Must use sequence parallelism if TP > 1

5. Is sequence length > 8K?
   Yes → Add CP (2-8)
   Use hierarchical CP for 32K+
```

### Constraints Checklist

Before finalizing configuration, verify:
- [ ] `TP divides num_attention_heads`
- [ ] `TP divides FFN_hidden_size`
- [ ] `PP divides num_layers`
- [ ] `CP divides sequence_length`
- [ ] `EP divides num_experts` (if MoE)
- [ ] `TP × PP × CP × EP × DP = Total GPUs`
- [ ] `num_microbatches >= 4 × PP` (for low bubble)
- [ ] `num_microbatches % PP == 0`
- [ ] Sequence parallelism enabled if `TP > 1 and EP > 1`

---

## NCCL Configuration and Tuning

### Environment Variables

**Critical for performance**:

```bash
# Kernel ordering (most important!)
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Ensures kernels launch in program order
# Required for proper communication-computation overlap

# NCCL optimizations
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1  # Specify IB adapters
export NCCL_SOCKET_IFNAME=eth0        # Fallback to Ethernet
export NCCL_DEBUG=INFO                # Debugging (disable in production)

# SHARP support (in-network reduction)
export NCCL_COLLNET_ENABLE=1          # Enable SHARP
# Only applies to first data-parallel group

# Network selection
export NCCL_NET=IB                    # Use InfiniBand
# Or: NCCL_NET=Socket for Ethernet
```

### Per-Group NCCL Configuration

Megatron allows **tuning NCCL per process group**:

```python
# From parallel_state.py - Example configuration
nccl_comm_cfgs = {
    "data_parallel": {
        "is_high_priority_stream": False,  # Low priority OK for DP
        "net_name": "IB",                   # Use InfiniBand
        "cga_cluster_size": 2,              # NCCL kernel tuning
    },
    "tensor_parallel": {
        "is_high_priority_stream": True,   # High priority! TP is latency-sensitive
        "net_name": "IB",
        "max_ctas": 32,                     # Max CTAs for NCCL kernel
    },
    "pipeline_parallel": {
        "is_high_priority_stream": False,
        "net_name": "Socket",               # Can use Ethernet for PP
    },
}
```

**When to tune**:
- TP: Latency-critical, use high priority
- DP: Bandwidth-bound, can tolerate latency
- PP: Large messages, less sensitive to tuning

### Network Topology Awareness

**NVLink vs InfiniBand**:
- **Intra-node**: Use NVLink (900 GB/s on H100)
- **Inter-node**: Use InfiniBand (400 Gb/s = 50 GB/s per link)

**Rank ordering**: Match parallelism to topology
```bash
--rank-ordering tp-cp-ep-dp-pp
# TP,CP,EP within node (use NVLink)
# DP,PP across nodes (use InfiniBand)
```

### SHARP (Scalable Hierarchical Aggregation)

**What is SHARP**: In-network reduction offloads all-reduce to the switch fabric.

**Benefits**:
- 50-70% reduction in CPU overhead
- Higher effective bandwidth
- Frees GPU SMs for computation

**Configuration**:
```bash
export NCCL_COLLNET_ENABLE=1
# Automatically used for first DP group if hardware supports
```

**Hardware requirements**:
- NVIDIA Quantum InfiniBand switches
- Mellanox ConnectX-6 or newer NICs

---

## Configuration Guidelines

### Step-by-Step Configuration

**1. Determine model size**:
- Small (< 13B): Likely fits in single GPU with TP=1
- Medium (13B-70B): Needs TP, maybe PP
- Large (70B-405B): Needs TP + PP
- MoE: Also needs EP

**2. Calculate memory requirements**:
```python
# Model memory (FP16):
model_params = num_layers * (hidden_size * ffn_hidden_size * 12 + hidden_size^2 * 4)
model_memory_gb = model_params * 2 / (1024^3)

# Optimizer memory (Adam FP32):
optimizer_memory_gb = model_memory_gb * 8  # 2× for momentum, variance, master weights

# Activation memory (per microbatch):
activation_memory_gb = num_layers * batch_size * seq_len * hidden_size * 34 * 2 / (1024^3)
# Factor of 34 is approximation for transformer activations
```

**3. Select TP size**:
```python
if model_memory_gb > gpu_memory_gb:
    tp_size = ceil(model_memory_gb / gpu_memory_gb)
    tp_size = next_power_of_2(tp_size)  # Must be power of 2
    tp_size = min(tp_size, 8)  # Usually don't exceed 8
else:
    tp_size = 1
```

**4. Select PP size**:
```python
remaining_memory_needed = model_memory_gb / tp_size + optimizer_memory_gb / tp_size
if remaining_memory_needed > gpu_memory_gb:
    pp_size = ceil(remaining_memory_needed / gpu_memory_gb)
else:
    pp_size = 1
```

**5. Select CP size** (if long sequences):
```python
if sequence_length > 8192:
    cp_size = 2
if sequence_length > 32768:
    cp_size = 4
```

**6. Select EP size** (if MoE):
```python
ep_size = num_experts // max_experts_per_gpu
```

**7. Calculate DP size**:
```python
dp_size = total_gpus // (tp_size * pp_size * cp_size * ep_size)
```

### Common Configurations

**Single Node (8× H100 80GB)**:

Small models (7B-13B):
```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
# DP = 8
```

Medium models (30B-70B):
```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 1
--sequence-parallel
# DP = 2
```

Large models (70B+ with long sequences):
```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 2
--sequence-parallel
# DP = 1
```

**Multi-Node (128× H100, 16 nodes)**:

Medium models (70B):
```bash
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 4
--virtual-pipeline-model-parallel-size 2
--sequence-parallel
# DP = 4
```

Large models (175B-405B):
```bash
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 16
--virtual-pipeline-model-parallel-size 4
--sequence-parallel
# DP = 1
```

MoE models (Mixtral-style, 8×7B):
```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 8
--expert-model-parallel-size 8
--num-experts 8
--moe-grouped-gemm
# DP = 2
```

### Troubleshooting

**Symptom**: Low GPU utilization (< 30%)
- **Cause**: Likely high pipeline bubbles or communication overhead
- **Solution**: Increase num_microbatches, enable virtual PP, check overlap flags

**Symptom**: OOM (Out of Memory)
- **Cause**: Model too large for current parallelism config
- **Solution**: Increase TP or PP, enable activation checkpointing, use FSDP

**Symptom**: Slow all-reduce
- **Cause**: Suboptimal DP group configuration
- **Solution**: Check NCCL env vars, enable gradient bucketing, use overlap_grad_reduce

**Symptom**: Uneven GPU load (some GPUs idle)
- **Cause**: Load imbalance in MoE routing
- **Solution**: Enable aux-loss-free balancing, check expert assignment distribution

---

## Performance Impact

### Scaling Efficiency

**Near-linear scaling** has been demonstrated:
- **LLaMA-3 70B**: 90% scaling efficiency from 8 to 64 GPUs
- **LLaMA-3.1 405B**: 85% scaling efficiency from 64 to 512 GPUs
- **DeepSeek-V3 671B**: Trained on 1024 GPUs with 2.64M tokens/sec

### GPU Utilization

**Model FLOP Utilization (MFU)**:
- H100: **47% MFU** achieved on large-scale training
- A100: **35-40% MFU** on similar workloads

**For comparison**:
- Theoretical peak: 100% (never achievable in practice)
- Good performance: 40-50%
- Megatron's 47%: State-of-the-art

### Communication Overhead

With proper configuration, communication can be almost entirely hidden:
- **Gradient reduction**: 80-90% overlap (DP)
- **Parameter gather**: 70-80% overlap (distributed optimizer)
- **TP all-reduce**: 60-70% overlap (sequence parallelism + async ops)
- **PP P2P**: Minimal overhead with sufficient microbatches

### Pipeline Bubble Analysis

**Quantitative comparison**:

| Schedule | Bubble Time | Relative Efficiency |
|----------|-------------|---------------------|
| GPipe (naive) | (p-1)/p | ~87% wasted (p=8) |
| 1F1B | (p-1)/(m+p-1) | ~18% wasted (p=8, m=32) |
| 1F1B Interleaved | ~(p-1)/(m×v+p-1) | ~9% wasted (p=8, m=32, v=2) |

**For p=8, m=64, v=4**:
- GPipe: 87.5% bubble
- 1F1B: 9.9% bubble
- 1F1B Interleaved: 2.7% bubble ← **Nearly optimal!**

---

## References and Further Reading

**Key Source Files**:
- `megatron/core/parallel_state.py:1-2687` - Process group management
- `megatron/core/tensor_parallel/layers.py:1-1425` - Tensor parallelism implementation
- `megatron/core/pipeline_parallel/schedules.py:1-2800` - Pipeline scheduling
- `megatron/core/distributed/param_and_grad_buffer.py:1-1007` - Gradient bucketing
- `megatron/core/optimizer/distrib_optimizer.py:1-3264` - Distributed optimizer

**Related Documentation**:
- [Communication Overlap](02-communication-overlap.md) - Detailed analysis of overlap techniques
- [Pipeline Scheduling](03-pipeline-scheduling.md) - Deep dive into 1F1B and interleaved schedules
- [Memory Optimizations](06-memory-optimizations.md) - Memory efficiency through FSDP and activation checkpointing

**External Resources**:
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Original tensor parallelism paper
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473) - Pipeline parallelism
- [ZeRO Paper](https://arxiv.org/abs/1910.02054) - Memory-efficient data parallelism

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
**Part of**: Megatron-LM GPU Utilization Analysis Series
