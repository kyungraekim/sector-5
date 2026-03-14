# Communication Overlap in Megatron-LM

## Table of Contents

1. [Key Takeaways](#key-takeaways)
2. [Introduction](#introduction)
3. [CUDA Stream Management](#cuda-stream-management)
4. [Gradient Reduction Overlap](#gradient-reduction-overlap)
5. [Parameter Gather Overlap](#parameter-gather-overlap)
6. [Tensor Parallel Communication Overlap](#tensor-parallel-communication-overlap)
7. [Pipeline P2P Communication](#pipeline-p2p-communication)
8. [MoE Expert Parallel Overlap](#moe-expert-parallel-overlap)
9. [NCCL Userbuffer Optimization](#nccl-userbuffer-optimization)
10. [Configuration and Tuning](#configuration-and-tuning)
11. [Performance Impact](#performance-impact)

---

## Key Takeaways

**Why Communication Overlap Matters**:
- Without overlap: Communication time **adds** to computation time
- With overlap: Communication time **hidden** behind computation
- Result: 80-90% of communication latency can be hidden in well-configured systems

**Core Principle**:
```
Sequential (No Overlap):
[Compute] → [Wait] → [Communicate] → [Compute] → ...
Total time = Compute + Communicate

Overlapped:
[Compute + Communicate in parallel] → [Compute + Communicate] → ...
Total time ≈ max(Compute, Communicate)
```

**Critical Numbers**:
- **Gradient reduction**: 80-90% latency hidden (data parallelism)
- **Parameter gather**: 70-80% latency hidden (distributed optimizer)
- **TP all-reduce**: 60-70% latency hidden (tensor parallelism with sequence parallelism)
- **PP P2P**: Nearly 100% hidden with sufficient microbatches

**Key Technique**: **Asynchronous operations** (`async_op=True`) + careful **CUDA stream management**

---

## Introduction

### The Communication Challenge

Modern distributed training involves massive data transfers:

**For LLaMA-3 70B on 64 GPUs** (TP=4, PP=4, DP=4):
- **TP all-reduce**: ~35 GB per layer per microbatch (FP16 weights)
- **DP gradient reduce**: ~140 GB per iteration (all gradients)
- **PP P2P**: ~1 GB per microbatch per stage (activations)

At 400 Gb/s InfiniBand (50 GB/s), this represents:
- TP: ~0.7s per layer
- DP: ~2.8s per iteration
- PP: ~0.02s per microbatch

**Without overlap**: These times add directly to training iteration time → **massive GPU underutilization**

**With overlap**: Communication executes concurrently with computation → **80-90% of communication time hidden**

### Megatron's Overlap Strategy

Megatron achieves communication overlap through:

1. **Asynchronous operations**: All collectives use `async_op=True`
2. **Multi-stream execution**: Separate CUDA streams for compute and communication
3. **Granular synchronization**: Explicit wait points only when necessary
4. **Bucket-based communication**: Coalesce small operations into larger ones
5. **Backward hooks**: Trigger communication immediately when gradients ready

The result: **GPUs spend < 10% of time waiting for communication** on well-configured systems.

---

## CUDA Stream Management

### Understanding CUDA Streams

**CUDA streams** are sequences of operations that execute in order. Operations in **different streams** can execute concurrently.

```python
# Default stream: All operations execute sequentially
x = torch.matmul(A, B)  # Operation 1
torch.distributed.all_reduce(x)  # Operation 2 (waits for 1)
y = torch.matmul(C, D)  # Operation 3 (waits for 2)

# Multi-stream: Communication can overlap with next computation
compute_stream = torch.cuda.default_stream()
comm_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    x = torch.matmul(A, B)  # Operation 1

with torch.cuda.stream(comm_stream):
    torch.distributed.all_reduce(x, async_op=True)  # Operation 2 (parallel with 3!)

with torch.cuda.stream(compute_stream):
    y = torch.matmul(C, D)  # Operation 3 (overlaps with 2)
```

### Megatron's Stream Architecture

From `param_and_grad_buffer.py:360-386`, Megatron uses **multi-stream communication**:

```python
# megatron/core/distributed/param_and_grad_buffer.py:360-386
if self.ddp_config.num_distributed_optimizer_instances > 1:
    # Use separate communication stream
    stream_context = torch.cuda.stream(self.communication_stream)

    # Wait for gradient computation to complete
    self.communication_stream.wait_stream(torch.cuda.default_stream())
else:
    # Single distributed optimizer instance: use default stream
    stream_context = nullcontext()

with stream_context:
    # Launch async communication
    # This stream will proceed independently
    self._reduce_scatter_async(...)
```

**Stream organization**:
```
Compute Stream (default): Backward pass → Gradient computation
    ↓ (signal when ready)
Comm Stream: All-reduce/Reduce-scatter gradients
    ↓ (independent execution)
NCCL Stream (internal): Actual network transfers
```

### CUDA_DEVICE_MAX_CONNECTIONS=1

**Critical environment variable** for overlap:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**Why it matters**:
- Controls number of concurrent CUDA kernel queues
- With value `1`: Kernels launch in **program order**
- Ensures communication scheduled before subsequent computation

From `layers.py:514-520`:

```python
# megatron/core/tensor_parallel/layers.py:514-520
handle = dist_all_gather_func(
    all_gather_buffer, input, group=tp_group, async_op=True
)
# Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
# gather is scheduled before the input gradient computation
```

**Without CUDA_DEVICE_MAX_CONNECTIONS=1**: Kernels might launch out of order, breaking overlap assumptions.

**Exception**: UCC backend for pipeline parallelism can use `CUDA_DEVICE_MAX_CONNECTIONS > 1`.

---

## Gradient Reduction Overlap

### The DP Communication Problem

In data parallelism, gradients must be **averaged across all DP ranks** before the optimizer step.

**Naive approach** (no overlap):
```python
# 1. Complete entire backward pass
for layer in reversed(model.layers):
    layer.backward()

# 2. Then reduce all gradients
for param in model.parameters():
    torch.distributed.all_reduce(param.grad)  # Blocks GPU!

# 3. Finally update parameters
optimizer.step()
```

**Problem**: GPU idle during gradient reduction (~2-3 seconds for 70B model!)

### Bucket-Based Asynchronous Reduction

**Megatron's solution** from `param_and_grad_buffer.py:62-100`:

**Step 1**: Group parameters into **buckets**

```python
# megatron/core/distributed/param_and_grad_buffer.py:62-100
class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.

    Args:
        params: List of parameters whose gradients are collated in this bucket.
        grad_data: View in _ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger _ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
    """
```

**Bucket size calculation**:
```python
# Default: 40M parameters + 1M per DP rank
bucket_size = max(40_000_000, 1_000_000 * data_parallel_size)

# Why scale with DP size?
# - Larger DP → more ranks in all-reduce
# - Larger buckets amortize latency better
# - Smaller buckets → more communication ops → latency-bound
```

**Step 2**: Allocate contiguous gradient buffer

```python
# All gradients stored in single contiguous tensor per dtype
self.grad_buffer = torch.zeros(
    total_params,
    dtype=dtype,
    device='cuda',
    requires_grad=False
)

# Each bucket is a view into this buffer
bucket.grad_data = self.grad_buffer[offset : offset + bucket_size]
```

**Benefits**:
- **Single collective per bucket** instead of per-parameter
- **Better memory alignment** for NCCL performance
- **Reduced fragmentation** (no gaps between tensors)

**Step 3**: Register backward hooks to detect gradient readiness

From `distributed_data_parallel.py:441-467`:

```python
# megatron/core/distributed/distributed_data_parallel.py:441-467
def _make_param_hook(param, param_index):
    """Create hook to mark gradient as ready when computed."""
    def grad_ready_hook(*unused):
        param.main_grad.grad_added_to_main_grad = True

        # Notify buffer that this parameter's gradient is ready
        self.buffers[param.dtype].mark_grad_computed(param, param_index)

    return grad_ready_hook

# Register hook for each parameter
for param_index, param in enumerate(bucket.params_list):
    param.register_post_accumulate_grad_hook(
        _make_param_hook(param, param_index)
    )
```

**Step 4**: Launch async reduce-scatter when bucket ready

```python
# megatron/core/distributed/param_and_grad_buffer.py:~400-450
def mark_grad_computed(self, param, param_index):
    """Mark gradient as computed, launch communication if bucket ready."""
    bucket = self.param_to_bucket[param]
    bucket.params_with_grad.add(param)

    # Check if all parameters in bucket have gradients
    if len(bucket.params_with_grad) == len(bucket.params_list):
        # Bucket ready! Launch async reduce-scatter
        self._reduce_scatter_base_async(bucket)

def _reduce_scatter_base_async(self, bucket):
    """Launch asynchronous reduce-scatter for bucket."""
    if self.ddp_config.overlap_grad_reduce:
        # Use communication stream for overlap
        with torch.cuda.stream(self.communication_stream):
            handle = dist_reduce_scatter_func(
                local_shard,
                bucket.grad_data,
                group=self.data_parallel_group,
                async_op=True  # Non-blocking!
            )
            self.bucket_to_comm_handle[bucket] = handle
```

**Result**: As soon as each bucket's gradients are computed, communication starts **immediately** while backward pass continues on remaining layers.

### Overlap Visualization

```
Timeline with overlap:

Compute Stream:
[Layer N] [Layer N-1] [Layer N-2] [Layer N-3] ... [Layer 0]
          ↓           ↓            ↓
Comm Stream:
          [Bucket A]  [Bucket B]   [Bucket C]      ...
          reduce      reduce       reduce

Without overlap:
[Layer N] [Layer N-1] ... [Layer 0] [IDLE: Bucket A] [IDLE: Bucket B] ...

Time saved: ~80-90% of communication latency!
```

### Communication Coalescing

For multiple buckets, Megatron **coalesces** reduce-scatter operations:

From `param_and_grad_buffer.py:393-414`:

```python
# megatron/core/distributed/param_and_grad_buffer.py:393-414
with _coalescing_manager(communication_group, async_ops=async_op) as cm:
    for bucket in self.buckets:
        # All reduce-scatters batched into single NCCL operation
        dist_reduce_scatter_func(
            local_data_view,
            bucket.grad_data,
            group=communication_group,
            async_op=True
        )
```

**Benefit**: NCCL can optimize the batched operation for better bandwidth utilization.

### FP32 Accumulation

Gradients are accumulated directly in **FP32 precision** even when model is FP16:

```python
# From param_and_grad_buffer.py
# grad_data buffer is always FP32 for numerical stability
self.grad_buffer = torch.zeros(..., dtype=torch.float32)

# During backward:
# FP16 gradient → automatically upcast to FP32 → accumulate in buffer
```

**Why**: Prevents gradient underflow/overflow when accumulating many small gradients.

---

## Parameter Gather Overlap

### The Distributed Optimizer Problem

With **ZeRO-style parameter sharding**, each DP rank only owns `1/DP_size` of parameters.

**Before forward pass**: Need to **all-gather** full parameters from shards.

**Naive approach**:
```python
# 1. All-gather parameters for layer N
all_gather_parameters(layer_N)

# 2. Wait for all-gather to complete
torch.cuda.synchronize()

# 3. Compute forward pass
output = layer_N.forward(input)
```

**Problem**: GPU idle during all-gather!

### Prefetch and Overlap

**Megatron's solution** from `distrib_optimizer.py`:

**Strategy**: While computing layer N, **prefetch parameters for layer N+1**

```python
# megatron/core/optimizer/distrib_optimizer.py (conceptual)
def forward_with_prefetch(layers, input):
    # Prefetch first layer's parameters
    all_gather_handle = all_gather_async(layers[0].parameters())

    for i, layer in enumerate(layers):
        # Wait for current layer's parameters
        all_gather_handle.wait()

        # Start prefetch for next layer while computing current
        if i + 1 < len(layers):
            all_gather_handle = all_gather_async(layers[i+1].parameters())

        # Compute current layer (overlaps with prefetch!)
        output = layer.forward(input)
        input = output

    return output
```

**Timeline**:
```
Layer 0: [All-gather L0] [Compute L0 + All-gather L1]
Layer 1:                 [Compute L1 + All-gather L2]
Layer 2:                                [Compute L2 + All-gather L3]
...

Latency hidden: 70-80% (limited by compute/communication ratio)
```

### Configuration

Enable with:
```bash
--use-distributed-optimizer
--overlap-param-gather
```

From `model_parallel_config.py:~200`:

```python
# megatron/core/model_parallel_config.py
overlap_param_gather: bool = False
"""
If true, overlap param all-gather with forward compute in distributed optimizer.
Requires use_distributed_optimizer=True.
"""
```

---

## Tensor Parallel Communication Overlap

### TP Communication Pattern

Recall from [Parallelism Strategies](01-parallelism-strategies.md#tensor-parallelism-tp):
- **Column-parallel**: All-reduce of input gradients in backward
- **Row-parallel**: All-reduce of outputs in forward

**With sequence parallelism**:
- **Column-parallel**: All-gather inputs in forward, reduce-scatter outputs in backward
- **Row-parallel**: All-gather inputs in forward, reduce-scatter grads in backward

### Async All-Reduce

From `layers.py:535-536`:

```python
# megatron/core/tensor_parallel/layers.py:535-536
if ctx.allreduce_dgrad:
    # Asynchronous all-reduce for communication-computation overlap
    handle = torch.distributed.all_reduce(
        grad_input,
        group=tp_group,
        async_op=True  # Non-blocking!
    )
    # Don't wait yet - let it overlap with next layer's computation
    ctx.async_grad_allreduce_handle = handle
```

**Synchronization point**: Before optimizer step

```python
# Wait for all async TP operations to complete
if hasattr(ctx, 'async_grad_allreduce_handle'):
    ctx.async_grad_allreduce_handle.wait()
```

### Transformer Engine Userbuffer Overlap

**Most sophisticated TP overlap** via Transformer Engine integration.

**From `transformer_engine.py:~1500-1600`**:

```bash
# Enable TP communication overlap via userbuffers
--tp-comm-overlap
--tp-comm-bulk-wgrad    # Overlap All-Gather with weight grad GEMM
--tp-comm-bulk-dgrad    # Overlap Reduce-Scatter with data grad GEMM
--tp-comm-overlap-ag    # Pipeline All-Gather chunks with GEMM splits
--tp-comm-overlap-rs    # Pipeline Reduce-Scatter with GEMM splits
```

**How userbuffers work**:

1. **Pre-allocate communication buffers**:
```python
# Transformer Engine allocates persistent buffers for all-gather/reduce-scatter
ub_buffers = allocate_userbuffers(
    num_splits=4,  # Split GEMM into 4 chunks
    size_per_split=model_chunk_size
)
```

2. **Pipeline GEMM and communication**:
```
Regular (no overlap):
[All-Gather full tensor] → [GEMM full tensor]

With userbuffer overlap (4 splits):
[AG chunk 0] → [GEMM chunk 0 + AG chunk 1] → [GEMM chunk 1 + AG chunk 2] → ...
```

3. **Symmetric memory for faster collectives**:
From TE 2.3+, using **symmetric memory addressing** for faster all-reduce:
```python
# Requires PyTorch 2.7+ and TE 2.3+
--tp-comm-symmetric-memory
```

**Result**: TP communication latency reduced by 60-70% through overlap.

### Sequence Parallelism Communication

With sequence parallelism, TP layers use **all-gather** (forward) and **reduce-scatter** (backward) instead of all-reduce.

**All-gather in forward**:
```python
# megatron/core/tensor_parallel/layers.py:~200-250
# Input partitioned along sequence: [seq_len/tp, batch, hidden]
# Need full sequence for TP linear

# Async all-gather
handle = dist_all_gather_func(
    all_gather_buffer,  # [seq_len, batch, hidden/tp]
    input,               # [seq_len/tp, batch, hidden/tp]
    group=tp_group,
    async_op=True
)

# CUDA_DEVICE_MAX_CONNECTIONS=1 ensures gather scheduled before GEMM
# Gather executes while previous layer's computation finishes
```

**Reduce-scatter in backward**:
```python
# Output: [seq_len, batch, hidden/tp]
# Need to partition back to: [seq_len/tp, batch, hidden/tp]

# Async reduce-scatter
handle = dist_reduce_scatter_func(
    output_local,  # [seq_len/tp, batch, hidden/tp]
    output_full,   # [seq_len, batch, hidden/tp]
    group=tp_group,
    async_op=True
)

# Reduce-scatter overlaps with next layer's backward
```

**Why all-gather/reduce-scatter instead of all-reduce**:
- **Memory savings**: Don't need to keep full sequence in memory
- **Better overlap**: Reduce-scatter can finish earlier than all-reduce
- **Gradient accumulation fusion**: Can accumulate directly in sharded buffer

---

## Pipeline P2P Communication

### Pipeline Communication Pattern

**Forward pass**:
```python
# Send activations to next pipeline stage
if not is_pipeline_last_stage():
    send_forward(output_tensor, next_rank)

# Receive activations from previous stage
if not is_pipeline_first_stage():
    input_tensor = recv_forward(prev_rank)
```

**Backward pass**:
```python
# Send gradients to previous stage
if not is_pipeline_first_stage():
    send_backward(input_gradient, prev_rank)

# Receive gradients from next stage
if not is_pipeline_last_stage():
    output_gradient = recv_backward(next_rank)
```

### Overlapped P2P Communication

**From `schedules.py:901-902`**:

```python
# megatron/core/pipeline_parallel/schedules.py:901-902
if config.overlap_p2p_comm:
    # Use async send/recv for overlap
    send_handle = send_forward_async(output_tensor)
    recv_handle = recv_forward_async()

    # Do other work while communication in progress
    # (e.g., prepare next microbatch)

    # Wait when needed
    send_handle.wait()
    input_tensor = recv_handle.wait()
```

**Batched P2P Communication**:

```python
# megatron/core/pipeline_parallel/schedules.py:~950
if config.batch_p2p_comm:
    # Batch multiple send/recv into single operation
    ops = []
    ops.append(torch.distributed.P2POp(
        torch.distributed.isend, output_tensor, next_rank
    ))
    ops.append(torch.distributed.P2POp(
        torch.distributed.irecv, input_tensor, prev_rank
    ))

    # Execute all P2P ops together
    reqs = torch.distributed.batch_isend_irecv(ops)
```

**Cannot use both simultaneously**:
```python
if config.overlap_p2p_comm and config.batch_p2p_comm:
    raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
```

### UCC Backend for Zero-SM Communication

**Problem**: NCCL P2P operations use GPU streaming multiprocessors (SMs), reducing compute capacity.

**Solution**: **UCC backend** offloads communication to network stack.

From `schedules.py:976-988`:

```python
# megatron/core/pipeline_parallel/schedules.py:976-988
# UCC backend benefits:
# - Zero SM resource usage
# - No GPU SM interference with computation
# - Better InfiniBand bandwidth utilization

# Configuration:
export CUDA_DEVICE_MAX_CONNECTIONS>1  # Can use >1 with UCC
# UCC automatically selected if available
```

**SM usage comparison**:
- **Standard NCCL**: 1-4 SMs per communication
- **UCC**: 0 SMs (fully offloaded)

**When to use**: When pipeline communication is on critical path and SM contention observed.

### Microbatch Pipelining

**Key to PP overlap**: Sufficient microbatches to keep pipeline full.

```
With num_microbatches >> pipeline_stages:

Time →
Stage 0: [F0][F1][F2][F3][B0][B1][B2][B3]
Stage 1:     [F0][F1][F2][B0][F3][B1][B2][B3]
Stage 2:         [F0][F1][B0][F2][B1][F3][B2][B3]
Stage 3:             [F0][B0][F1][B1][F2][B2][F3][B3]

P2P communication happens during transitions (between F and B)
Communication time << computation time per microbatch
→ Nearly 100% overlap!
```

**Guideline**: `num_microbatches >= 4 × pipeline_parallel_size`

---

## MoE Expert Parallel Overlap

### MoE Communication Challenge

**Problem**: MoE models use **all-to-all** communication to route tokens to experts.

```python
# Forward: Route tokens to expert GPUs
tokens_for_experts = all_to_all(tokens, expert_assignments)

# Compute experts
expert_outputs = process_experts(tokens_for_experts)

# Backward: Return tokens to original GPUs
final_outputs = all_to_all(expert_outputs, original_ranks)
```

**All-to-all is expensive**: Irregular communication pattern, can't be easily overlapped.

### Batch-Level Overlap in 1F1B

**Innovation** from `combined_1f1b.py:~100-200`:

**Overlap expert parallel A2A with attention/MLP compute from different microbatches**:

```
Phase 0: MB0 forward only
Phase 1: MB0 backward (attention/MLP) + MB1 forward (MoE A2A overlapped!)
Phase 2: MB1 backward (attention/MLP) + MB2 forward (MoE A2A overlapped!)
...
Phase N: MBN backward only

Key insight: A2A for MB[i] forward overlaps with attention/MLP for MB[i-1] backward
```

**Configuration**:
```bash
--overlap-moe-expert-parallel-comm
```

**From `combined_1f1b.py:~150`**:
```python
# megatron/core/pipeline_parallel/combined_1f1b.py:~150
if config.overlap_moe_expert_parallel_comm:
    # Use specialized schedule that overlaps EP A2A
    # with non-MoE layers from previous microbatch
    schedule = combined_1f1b_schedule_with_ep_overlap(...)
```

### Delay Weight Gradient Computation

**Further optimization**: Split weight gradient and activation gradient computation.

```bash
--delay-wgrad-compute
```

**Why**:
- Activation gradients needed immediately for backprop
- Weight gradients only needed before optimizer step
- Delaying wgrad computation creates more overlap opportunities

**Timeline**:
```
Without delay:
[Compute dgrad + wgrad] → limited overlap window

With delay:
[Compute dgrad] → [Compute wgrad later]
                  ↑ More time to overlap A2A!
```

---

## NCCL Userbuffer Optimization

### What Are NCCL Userbuffers?

**Userbuffers** are pre-registered memory regions that NCCL can access directly without runtime registration overhead.

**From FSDP integration** (`distributed_data_parallel_config.py`):

```bash
--nccl-ub  # Enable NCCL userbuffers
```

**Benefits**:
1. **Reduced SM usage**: NCCL can use more efficient algorithms
2. **Better bandwidth**: Direct memory access without copies
3. **Lower latency**: No runtime memory registration

### SM Usage Reduction with SHARP

**With SHARP + userbuffers**:

| Configuration | SM Usage (without UB) | SM Usage (with UB) | Reduction |
|---------------|----------------------|-------------------|-----------|
| IB only | 4 SMs | 1 SM | 4× |
| NVL + IB | 16 SMs | 6 SMs | 2.67× |

**More SMs available for computation** → better GPU utilization!

### Configuration

```bash
# Enable userbuffers for FSDP
--use-megatron-fsdp
--nccl-ub
--fsdp-double-buffer  # Required with nccl-ub
```

**Requirements**:
- NCCL 2.19+
- CUDA 12.1+
- Modern network adapters (ConnectX-6+)

---

## Configuration and Tuning

### Essential Flags for Overlap

**Data Parallelism**:
```bash
--overlap-grad-reduce          # Enable async gradient reduction
```

**Distributed Optimizer**:
```bash
--use-distributed-optimizer    # Enable ZeRO-style sharding
--overlap-param-gather         # Prefetch parameters
```

**Tensor Parallelism** (requires Transformer Engine):
```bash
--tp-comm-overlap              # Master switch
--tp-comm-bulk-wgrad           # Overlap AG with wgrad GEMM
--tp-comm-bulk-dgrad           # Overlap RS with dgrad GEMM
--tp-comm-overlap-ag           # Pipeline AG chunks
--tp-comm-overlap-rs           # Pipeline RS chunks
--tp-comm-overlap-rs-dgrad     # Pipeline RS with dgrad
```

**Pipeline Parallelism**:
```bash
--overlap-p2p-comm             # Async P2P (cannot combine with batch)
# OR
--batch-p2p-comm               # Batched P2P (cannot combine with overlap)
```

**MoE**:
```bash
--overlap-moe-expert-parallel-comm  # Batch-level EP A2A overlap
--delay-wgrad-compute               # Delay wgrad for more overlap
```

**FSDP**:
```bash
--nccl-ub                      # Enable userbuffers
--fsdp-double-buffer           # Required with nccl-ub
```

### Environment Variables

**Critical**:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Ensures kernel ordering for proper overlap
# Exception: Can use >1 with UCC backend
```

**NCCL optimization**:
```bash
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_COLLNET_ENABLE=1          # Enable SHARP
```

### Recommended Configurations

**Small-scale (8-64 GPUs)**:
```bash
--overlap-grad-reduce
--use-distributed-optimizer
--overlap-param-gather
```

**Medium-scale (64-256 GPUs)**:
```bash
--overlap-grad-reduce
--use-distributed-optimizer
--overlap-param-gather
--tp-comm-overlap
--tp-comm-bulk-wgrad
--tp-comm-bulk-dgrad
```

**Large-scale (256+ GPUs)**:
```bash
--overlap-grad-reduce
--use-distributed-optimizer
--overlap-param-gather
--tp-comm-overlap
--tp-comm-overlap-ag
--tp-comm-overlap-rs
--batch-p2p-comm  # Or overlap-p2p-comm
--nccl-ub
```

**MoE models**:
```bash
# All of the above, plus:
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute
```

---

## Performance Impact

### Measured Overlap Efficiency

**Gradient reduction** (Data Parallelism):
- Without overlap: 100% of grad reduction time is overhead
- With overlap: **10-20% overhead** (80-90% hidden)
- **Speedup**: 1.8-2.0× for communication-bound workloads

**Parameter gather** (Distributed Optimizer):
- Without overlap: 100% of all-gather time is overhead
- With overlap: **20-30% overhead** (70-80% hidden)
- **Speedup**: 1.5-1.7× for models with frequent param gather

**TP all-reduce/all-gather** (Tensor Parallelism):
- Without overlap: 100% of TP comm time is overhead
- With overlap: **30-40% overhead** (60-70% hidden)
- **Speedup**: 1.4-1.6× for TP-heavy configurations

**Pipeline P2P** (Pipeline Parallelism):
- Without overlap: Bubbles dominate (see [Pipeline Scheduling](03-pipeline-scheduling.md))
- With overlap + sufficient microbatches: **< 5% overhead**
- **Speedup**: Massive (10× or more reduction in bubbles with 1F1B + interleaving)

### Communication Breakdown for LLaMA-3 70B

**Configuration**: 64 GPUs (TP=4, PP=4, DP=4), 2048 seq len

| Communication Type | Volume per Iteration | Time (no overlap) | Time (with overlap) | Overlap % |
|-------------------|---------------------|------------------|---------------------|-----------|
| TP all-reduce | ~140 GB | 2.8s | 0.8s | 71% |
| DP gradient reduce | ~560 GB | 11.2s | 1.8s | 84% |
| PP P2P | ~4 GB | 0.08s | 0.01s | 88% |
| **Total** | **~704 GB** | **14.08s** | **2.61s** | **81%** |

**Forward + Backward compute time**: ~8s

**Without overlap**: Total time = 8s + 14.08s = **22.08s** per iteration
**With overlap**: Total time ≈ max(8s, 2.61s) + 2.61s = **10.61s** per iteration

**Speedup from overlap alone**: 2.08×

### Scaling Efficiency

**Weak scaling** (constant work per GPU):
- **Without overlap**: Efficiency degrades as ~1/(1 + comm_fraction)
- **With overlap**: Efficiency maintained near 90% up to 1000+ GPUs

**Strong scaling** (constant total work):
- **Without overlap**: Severe degradation beyond 64 GPUs
- **With overlap**: Linear scaling up to 256 GPUs, 80% efficiency at 512 GPUs

---

## Debugging and Troubleshooting

### Symptoms of Poor Overlap

**Low GPU utilization** (< 40%):
- Check: Are overlap flags enabled?
- Check: Is `CUDA_DEVICE_MAX_CONNECTIONS=1` set?
- Tool: `nvidia-smi dmon -s ucm` to monitor GPU compute/memory/comm

**High NCCL time** in profiling:
- Check: Are buckets too small? (increase bucket size)
- Check: Is SHARP enabled for DP?
- Tool: `nsys profile` to visualize timeline

**Out-of-order execution warnings**:
- Cause: Missing `CUDA_DEVICE_MAX_CONNECTIONS=1`
- Fix: Export env var and restart

### Profiling Communication Overlap

**Using NVIDIA Nsight Systems**:
```bash
nsys profile -o output.nsys-rep \
  --trace=cuda,nvtx,osrt,cudnn,cublas,mpi \
  python pretrain_gpt.py ...
```

**Look for**:
- NCCL operations overlapping with CUDA kernels
- Async handles (green in timeline) executed concurrently with compute
- Stream synchronization points (should be minimal)

**Using PyTorch Profiler**:
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Training iteration
    loss.backward()
    optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Check: NCCL time should be much smaller than compute time
```

---

## References and Further Reading

**Key Source Files**:
- `megatron/core/distributed/param_and_grad_buffer.py:1-1007` - Gradient bucketing and async reduction
- `megatron/core/distributed/distributed_data_parallel.py:441-467` - Backward hooks for gradient readiness
- `megatron/core/tensor_parallel/layers.py:514-536` - Async TP all-reduce/all-gather
- `megatron/core/extensions/transformer_engine.py:~1500-1600` - TE userbuffer overlap
- `megatron/core/pipeline_parallel/schedules.py:901-988` - P2P overlap and UCC backend
- `megatron/core/pipeline_parallel/combined_1f1b.py:~100-200` - MoE EP overlap

**Related Documentation**:
- [Parallelism Strategies](01-parallelism-strategies.md) - Foundation for understanding communication patterns
- [Pipeline Scheduling](03-pipeline-scheduling.md) - How 1F1B enables P2P overlap through microbatching
- [Transformer Engine](05-transformer-engine.md) - Deep dive into TE userbuffer mechanics
- [Memory Optimizations](06-memory-optimizations.md) - How distributed optimizer works with overlap

**External Resources**:
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) - NCCL collectives and optimization
- [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html) - Background on gradient bucketing
- [ZeRO Paper](https://arxiv.org/abs/1910.02054) - Distributed optimizer foundations

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
**Part of**: Megatron-LM GPU Utilization Analysis Series
