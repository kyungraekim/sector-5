# Pipeline Scheduling in Megatron-LM

> **Last Updated**: 2025-12-04
> **Related Documents**: [Parallelism Strategies](01-parallelism-strategies.md) | [Communication Overlap](02-communication-overlap.md)

## Table of Contents

1. [Overview](#overview)
2. [1F1B Schedule Fundamentals](#1f1b-schedule-fundamentals)
3. [Schedule Phases](#schedule-phases)
4. [Interleaved Pipeline (Virtual Pipeline Parallelism)](#interleaved-pipeline-virtual-pipeline-parallelism)
5. [Bubble Analysis](#bubble-analysis)
6. [P2P Communication](#p2p-communication)
7. [Schedule Selection](#schedule-selection)
8. [Implementation Deep Dives](#implementation-deep-dives)
9. [Performance Optimizations](#performance-optimizations)
10. [Configuration Guide](#configuration-guide)

---

## Overview

### What is Pipeline Scheduling?

Pipeline scheduling determines **when and where** to execute forward and backward passes across pipeline stages to maximize GPU utilization while minimizing memory usage.

**Core challenge**: Pipeline parallelism splits a model across N GPUs (stages), but naïve execution leads to **idle time (bubbles)** where most GPUs wait while one processes.

```
Naïve GPipe schedule (all forward, then all backward):
GPU 0: [F0] [F1] [F2] [F3] [idle........] [B3] [B2] [B1] [B0]
GPU 1: [idle] [F0] [F1] [F2] [F3] [idle..] [B3] [B2] [B1] [B0]
GPU 2: [idle....] [F0] [F1] [F2] [F3] [idle] [B3] [B2] [B1] [B0]
GPU 3: [idle........] [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0]

Bubble time: 75% idle! (Only 1/4 GPUs active most of the time)
```

**Solution**: **1F1B (One-Forward-One-Backward)** schedule interleaves forward and backward passes.

```
1F1B schedule:
GPU 0: [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
GPU 1: [idle] [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
GPU 2: [idle....] [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
GPU 3: [idle........] [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]

Bubble time: Much lower (~12.5% for PP=4, M=4)
```

### Key Concepts

**Microbatch**: Smallest unit of data processed in a single forward/backward pass. Global batch is split into M microbatches.

**Pipeline Stage (PP Rank)**: One section of the model on a single GPU. Total stages = PP size.

**Virtual Pipeline Stage (VP)**: Further splits each GPU's model chunk. Enables interleaved scheduling.

**Bubble**: Idle time when a GPU is waiting for dependencies (tensors from prev/next stage).

**Warmup Phase**: Initial forward-only steps to fill the pipeline.

**Steady-State (1F1B) Phase**: Alternating 1-forward-1-backward for memory efficiency.

**Cooldown Phase**: Final backward-only steps to drain the pipeline.

---

## 1F1B Schedule Fundamentals

### The 1F1B Pattern

**One-Forward-One-Backward** interleaves computation to keep all GPUs busy while limiting memory usage.

**Key insight**: After warmup, each GPU performs 1 forward pass followed immediately by 1 backward pass. This limits the number of "in-flight" activations to (PP_size - rank - 1), reducing peak memory.

### Basic 1F1B Flow

**Example**: PP=4 (4 GPUs), M=8 microbatches

```
Execution timeline (F = forward, B = backward):

GPU 0 (rank 0): F0 F1 F2 F3 F4 F5 F6 F7 | B0 B1 B2 B3 B4 B5 B6 B7
                |<-- warmup: 3 F -->|  |<-- steady state: 1F1B -->||<cooldown>|

GPU 1 (rank 1):    F0 F1 F2 F3 F4 F5 F6 F7 | B0 B1 B2 B3 B4 B5 B6 B7
                   |<- warmup: 2 F->| |<--- steady state: 1F1B --->||cooldown|

GPU 2 (rank 2):       F0 F1 F2 F3 F4 F5 F6 F7 | B0 B1 B2 B3 B4 B5 B6 B7
                      |warmup: 1F| |<---- steady state: 1F1B ---->||cooldown|

GPU 3 (rank 3):          F0 F1 F2 F3 F4 F5 F6 F7 | B0 B1 B2 B3 B4 B5 B6 B7
                         |warmup:0| |<----- steady state: 1F1B ----->|  |
```

**Warmup calculation** (from `schedules.py:721`):
```python
num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
# GPU 0: 4 - 0 - 1 = 3 warmup forwards
# GPU 1: 4 - 1 - 1 = 2 warmup forwards
# GPU 2: 4 - 2 - 1 = 1 warmup forward
# GPU 3: 4 - 3 - 1 = 0 warmup forwards
```

**Memory benefit**: At steady-state, each GPU holds activations for only (warmup + 1) microbatches, not all M microbatches.

---

## Schedule Phases

### Phase 1: Warmup

**Goal**: Fill the pipeline with forward passes so backward passes can start.

**Characteristics**:
- Forward-only passes
- Number of warmup steps decreases by rank
- Activations are saved for later backward passes
- Memory usage increases during this phase

**From `schedules.py:2122-2148`**, warmup loop:

```python
# Run warmup forward passes
for i in range(num_warmup_microbatches):
    # Receive activation tensor from previous stage
    input_tensor = p2p_communicator.recv_forward(
        recv_tensor_shapes,
        is_pp_first_stage(p2p_communicator.pp_group)
    )

    # Forward pass
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        ...
    )

    # Send activation tensor to next stage
    p2p_communicator.send_forward(
        output_tensor,
        send_tensor_shapes,
        is_pp_last_stage(p2p_communicator.pp_group)
    )

    # Save for backward pass later
    if not forward_only:
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
```

**Memory tracking** during warmup:

```python
# Example: GPU 1 (rank 1, warmup=2), M=8 microbatches
# After warmup phase:
#   input_tensors = [input_0, input_1]      # 2 saved inputs
#   output_tensors = [output_0, output_1]  # 2 saved outputs
# Peak memory: 2 × activation_size (vs M=8 × activation_size in GPipe)
```

### Phase 2: Steady-State (1F1B)

**Goal**: Maintain constant memory by doing 1 forward, then 1 backward.

**Characteristics**:
- Alternates forward and backward passes
- Memory usage stays constant (warmup + 1 microbatches)
- Maximum GPU utilization (all stages busy)
- Activations are reused immediately after backward

**Pattern**:
```python
for i in range(num_microbatches_remaining):
    # 1 Forward pass (for microbatch warmup + i + 1)
    forward_step(...)

    # 1 Backward pass (for microbatch i)
    backward_step(...)
```

**Memory stays constant**:
```python
# Before: holding warmup microbatches
# Add 1 forward: warmup + 1 microbatches
# Remove 1 backward: warmup microbatches again
# Result: Memory oscillates between warmup and warmup+1
```

### Phase 3: Cooldown

**Goal**: Complete all remaining backward passes.

**Characteristics**:
- Backward-only passes
- Drains the pipeline
- Memory usage decreases to zero
- Gradients are accumulated across all microbatches

**From `schedules.py:2180-2199`**, cooldown loop:

```python
# Run cooldown backward passes
for i in range(num_warmup_microbatches):
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    # Receive gradient from next stage
    output_tensor_grad = p2p_communicator.recv_backward(
        send_tensor_shapes,
        is_pp_last_stage(p2p_communicator.pp_group)
    )

    # Backward pass
    input_tensor_grad = backward_step(
        input_tensor,
        output_tensor,
        output_tensor_grad
    )

    # Send gradient to previous stage
    p2p_communicator.send_backward(
        input_tensor_grad,
        recv_tensor_shapes,
        is_pp_first_stage(p2p_communicator.pp_group)
    )
```

### Phase Transition Summary

| Phase | GPU 0 (rank 0) | GPU 1 (rank 1) | GPU 2 (rank 2) | GPU 3 (rank 3) |
|-------|---------------|---------------|---------------|---------------|
| **Warmup** | 3 F | 2 F | 1 F | 0 F |
| **Steady-State** | 5 × (1F+1B) | 6 × (1F+1B) | 7 × (1F+1B) | 8 × (1F+1B) |
| **Cooldown** | 3 B | 2 B | 1 B | 0 B |
| **Total Forwards** | 8 F | 8 F | 8 F | 8 F |
| **Total Backwards** | 8 B | 8 B | 8 B | 8 B |

---

## Interleaved Pipeline (Virtual Pipeline Parallelism)

### Concept

**Problem with standard 1F1B**: Bubble time = O(PP_size × time_per_microbatch).

**Solution**: Split each GPU's model into V "virtual stages" and interleave them.

**Virtual Pipeline Parallelism (VPP)** or **Interleaved Pipeline**: Each GPU holds V non-contiguous layer chunks and processes them in round-robin fashion.

### Model Chunking

**Standard PP=4** (no interleaving):
```
GPU 0: Layers 0-5
GPU 1: Layers 6-11
GPU 2: Layers 12-17
GPU 3: Layers 18-23
```

**Interleaved PP=4, VP=2**:
```
GPU 0: Chunk 0: Layers 0-2   + Chunk 1: Layers 12-14
GPU 1: Chunk 0: Layers 3-5   + Chunk 1: Layers 15-17
GPU 2: Chunk 0: Layers 6-8   + Chunk 1: Layers 18-20
GPU 3: Chunk 0: Layers 9-11  + Chunk 1: Layers 21-23
```

**Execution pattern**: Process chunk 0 for a few microbatches, then switch to chunk 1, back to chunk 0, etc.

### Schedule Table

**From `schedules.py:1038-1057`**, virtual pipeline schedule:

```python
# Example: PP=2, M=5 microbatches, VP=2, N=3 (microbatch_group_size_per_vp_stage)
# Total virtual microbatches = M × VP = 5 × 2 = 10

# Schedule table:
# virtual_microbatch_id | 0  1  2  3  4  5  6  7  8  9
# microbatch_id         | 0  1  2  0  1  2  3  4  3  4
# model_chunk_id        | 0  0  0  1  1  1  0  0  1  1

# Interpretation:
# - Virtual microbatches 0-2: Process microbatches 0-2 on chunk 0
# - Virtual microbatches 3-5: Process microbatches 0-2 on chunk 1
# - Virtual microbatches 6-7: Process microbatches 3-4 on chunk 0
# - Virtual microbatches 8-9: Process microbatches 3-4 on chunk 1
```

**Lookup helpers**:
```python
def get_model_chunk_id(virtual_microbatch_id, forward):
    """Get which model chunk to use for this virtual microbatch."""
    model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
    if not forward:
        # Backward processes chunks in reverse order
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id

def get_microbatch_id_in_model_chunk(iteration_id, forward):
    """Get the actual microbatch ID within the model chunk."""
    return microbatch_id_table[iteration_id]
```

### Warmup Calculation for Interleaved

**From `schedules.py:728-734`**, interleaved warmup:

```python
num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage

# Example: PP=4, rank=1, VP=2, N=3
# Base warmup: (4 - 1 - 1) * 2 = 4
# Virtual stages: (2 - 1) * 3 = 3
# Total warmup: 4 + 3 = 7 virtual microbatches
```

**Why more warmup?** Need to fill both virtual stages before starting backward passes.

### Bubble Reduction

**Standard 1F1B bubble time**:
```
Bubble fraction = (PP_size - 1) / M
# PP=4, M=8: 3/8 = 37.5% bubble
```

**Interleaved bubble time**:
```
Bubble fraction ≈ (PP_size - 1) / (M × VP)
# PP=4, M=8, VP=2: 3/16 = 18.75% bubble (2× reduction!)
# PP=4, M=8, VP=4: 3/32 = 9.4% bubble (4× reduction!)
```

**Trade-off**: More memory overhead (hold activations for more chunks) and context switching cost.

---

## Bubble Analysis

### Bubble Time Formula

**Bubble time** = Time when a GPU is idle waiting for dependencies.

**For standard 1F1B** (from Megatron-LM paper):

```
Total time = M × (T_f + T_b)  # M microbatches, each with forward (T_f) and backward (T_b)

Bubble time = (PP_size - 1) × (T_f + T_b)

Bubble fraction = Bubble time / Total time = (PP_size - 1) / M
```

**Example**: PP=8, M=16
```
Bubble fraction = 7 / 16 = 43.75%
```

**Guideline**: To keep bubble < 10%, need M > 10 × PP_size.

### Microbatch Selection

**Rule of thumb**:
```python
num_microbatches = 4 × PP_size  # Minimum for reasonable efficiency
num_microbatches = 8 × PP_size  # Good balance (12.5% bubble)
num_microbatches = 16 × PP_size  # Excellent (6.25% bubble)
```

**Constraints**:
- **Global batch size** = num_microbatches × micro_batch_size
- Larger microbatches → fewer microbatches → more bubbles
- Smaller microbatches → more microbatches → less efficient communication

### Memory vs. Bubble Trade-off

```
Memory usage per GPU ∝ num_warmup_microbatches
  = (PP_size - rank - 1) × activation_size

Bubble time ∝ (PP_size - 1) / num_microbatches
```

**Optimization strategy**:
1. **Increase M** (num_microbatches) to reduce bubbles
2. **Use interleaving** (VP > 1) to reduce bubbles without increasing M
3. **Enable activation checkpointing** to reduce memory from warmup microbatches

### Bubble Visualization

**Example: PP=4, M=8 microbatches**

```
Timeline (each cell = 1 time unit, T_f = T_b = 1):

Time: 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
GPU0: F0  F1  F2  F3  B0  F4  B1  F5  B2  F6  B3  F7  B4  B5  B6  B7
GPU1: [i] F0  F1  F2  B0  F3  B1  F4  B2  F5  B3  F6  B4  F7  B5  B6  B7
GPU2: [i] [i] F0  F1  B0  F2  B1  F3  B2  F4  B3  F5  B4  F6  B5  F7  B6  B7
GPU3: [i] [i] [i] F0  B0  F1  B1  F2  B2  F3  B3  F4  B4  F5  B5  F6  B6  F7  B7

Legend: Fi = forward microbatch i, Bi = backward microbatch i, [i] = idle (bubble)

Total time: 16 units
Bubble time (GPU 0): 0 units (0%)
Bubble time (GPU 1): 1 unit (6.25%)
Bubble time (GPU 2): 2 units (12.5%)
Bubble time (GPU 3): 3 units (18.75%)
Average bubble: (0+1+2+3)/4 / 16 = 9.4%

Theoretical: (PP-1)/M = 3/8 = 37.5%  (This is the bubble for the LAST stage)
Actual average: ~9.4% (much better due to earlier stages having less bubble)
```

---

## P2P Communication

### Point-to-Point Communication Primitives

Pipeline stages exchange tensors via **P2P send/recv** operations.

**From `p2p_communication.py:16-51`**, batched P2P:

```python
def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],  # Send to previous stage (backward direction)
    tensor_recv_prev: Optional[torch.Tensor],  # Receive from previous stage
    tensor_send_next: Optional[torch.Tensor],  # Send to next stage (forward direction)
    tensor_recv_next: Optional[torch.Tensor],  # Receive from next stage
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []

    # Build list of P2P operations
    if tensor_send_prev is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        ))
    if tensor_recv_prev is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        ))
    if tensor_send_next is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        ))
    if tensor_recv_next is not None:
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        ))

    # Launch all P2P ops in a single batch
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []

    return reqs
```

**Why batched?** Launching all send/recv together enables **overlap** and better network utilization.

### Forward Pass Communication

**Pattern**:
```python
# Receive activation from previous stage
input_tensor = recv_forward(from_prev_stage)

# Forward computation
output_tensor = model(input_tensor)

# Send activation to next stage
send_forward(output_tensor, to_next_stage)
```

**Edge cases**:
- **First stage** (rank 0): No recv, only send
- **Last stage** (rank PP-1): Recv, compute loss, no send

### Backward Pass Communication

**Pattern**:
```python
# Receive gradient from next stage
output_grad = recv_backward(from_next_stage)

# Backward computation
input_grad = backward(output_grad)

# Send gradient to previous stage
send_backward(input_grad, to_prev_stage)
```

**Edge cases**:
- **Last stage**: No recv (gradient initialized from loss)
- **First stage**: Backward compute, no send

### Even/Odd Rank Ordering

**From `p2p_communication.py:78-127`**, deadlock avoidance:

```python
if group.rank() % 2 == 0:
    # Even ranks: send-next, recv-prev, send-prev, recv-next
    if tensor_send_next is not None:
        send_next_req = torch.distributed.isend(...)
    if tensor_recv_prev is not None:
        recv_prev_req = torch.distributed.irecv(...)
    if tensor_send_prev is not None:
        send_prev_req = torch.distributed.isend(...)
    if tensor_recv_next is not None:
        recv_next_req = torch.distributed.irecv(...)
else:
    # Odd ranks: recv-prev, send-next, recv-next, send-prev
    if tensor_recv_prev is not None:
        recv_prev_req = torch.distributed.irecv(...)
    if tensor_send_next is not None:
        send_next_req = torch.distributed.isend(...)
    if tensor_recv_next is not None:
        recv_next_req = torch.distributed.irecv(...)
    if tensor_send_prev is not None:
        send_prev_req = torch.distributed.isend(...)
```

**Why different order?** Prevents **deadlock** when all ranks try to send before receiving.

**Example deadlock scenario** (if all ranks send first):
```
GPU 0 (even): Tries to send to GPU 1 → blocks waiting for GPU 1 to recv
GPU 1 (odd):  Tries to send to GPU 2 → blocks waiting for GPU 2 to recv
GPU 2 (even): Tries to send to GPU 3 → blocks waiting for GPU 3 to recv
GPU 3 (odd):  Tries to send to GPU 0 → blocks waiting for GPU 0 to recv
→ DEADLOCK! (All waiting for recv, none issuing recv)
```

**With even/odd ordering**:
```
GPU 0 (even): Sends to GPU 1 (async isend)
GPU 1 (odd):  Receives from GPU 0 (match! communication proceeds)
GPU 1 (odd):  Sends to GPU 2
GPU 2 (even): Receives from GPU 1 (match!)
... and so on (no deadlock)
```

### Tensor Shapes and Memory Allocation

**Activation tensor shape**:
```python
# Standard shape
[seq_length, micro_batch_size, hidden_size]

# With context parallelism
[seq_length / CP_size, micro_batch_size, hidden_size]

# With sequence parallelism
[seq_length / (TP_size × CP_size), micro_batch_size, hidden_size]
```

**Pre-allocation** (from `schedules.py:2090-2109`):
```python
# Calculate tensor shapes for recv/send
recv_tensor_shapes = get_tensor_shapes(
    seq_length=seq_length,
    micro_batch_size=micro_batch_size,
    decoder_seq_length=decoder_seq_length,
    config=config,
    tp_group=tp_group,
    cp_group=cp_group,
)

# Pre-allocate buffers for communication
# Avoids repeated allocation overhead during schedule execution
```

---

## Schedule Selection

### Decision Tree

**From `schedules.py:124-132`**, automatic selection:

```python
def get_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            # Interleaved 1F1B (with virtual pipeline)
            return forward_backward_pipelining_with_interleaving
        else:
            # Standard 1F1B (no interleaving)
            return forward_backward_pipelining_without_interleaving
    else:
        # No pipeline parallelism
        return forward_backward_no_pipelining
```

### When to Use Each Schedule

#### No Pipelining (PP=1)

**Use when**:
- Model fits on a single GPU
- Using TP, DP, or both (but not PP)

**Characteristics**:
- No bubbles (no pipeline)
- Simple forward-backward loop
- No P2P communication overhead

```bash
# No PP
--pipeline-model-parallel-size 1
```

#### Standard 1F1B (PP > 1, VP = None)

**Use when**:
- Model is too large for single GPU
- Moderate PP size (2-8)
- Sufficient microbatches (M > 8 × PP)

**Characteristics**:
- Memory-efficient (warmup + 1 activations)
- Bubble time: ~(PP-1)/M
- Simpler implementation

```bash
# Standard 1F1B
--pipeline-model-parallel-size 4  # PP=4
# No virtual-pipeline flag
```

#### Interleaved 1F1B (PP > 1, VP > 1)

**Use when**:
- Large PP size (8+)
- Need to reduce bubbles
- Have extra memory for multiple chunks

**Characteristics**:
- Lower bubble time: ~(PP-1)/(M×VP)
- Higher memory usage (VP chunks)
- More complex scheduling

```bash
# Interleaved 1F1B
--pipeline-model-parallel-size 8  # PP=8
--virtual-pipeline-model-parallel-size 4  # VP=4
```

### Configuration Guidelines

**For PP=4**:
```bash
# Minimum viable
--num-micro-batches 16  # M = 4 × PP, bubble ~18.75%

# Good balance
--num-micro-batches 32  # M = 8 × PP, bubble ~9.4%

# Optimal
--num-micro-batches 64  # M = 16 × PP, bubble ~4.7%
```

**For PP=8 with high bubble**:
```bash
# Option 1: Increase microbatches
--num-micro-batches 128  # M = 16 × PP, bubble ~4.3%

# Option 2: Use interleaving
--virtual-pipeline-model-parallel-size 2  # VP=2, bubble cut in half
--num-micro-batches 64  # M = 8 × PP, effective M×VP = 128
```

---

## Implementation Deep Dives

### Standard 1F1B Implementation

**From `schedules.py:2068-2074`**, warmup microbatch calculation:

```python
# Compute number of warmup microbatches
num_warmup_microbatches = (
    p2p_communicator.pp_group.size() - p2p_communicator.pp_group.rank() - 1
)
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches

# Example: PP=4, M=8
# Rank 0: warmup=3, remaining=5
# Rank 1: warmup=2, remaining=6
# Rank 2: warmup=1, remaining=7
# Rank 3: warmup=0, remaining=8
```

**Warmup phase** (forward-only):

```python
for i in range(num_warmup_microbatches):
    input_tensor = p2p_communicator.recv_forward(
        recv_tensor_shapes,
        is_pp_first_stage(p2p_communicator.pp_group)
    )

    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        ...
    )

    p2p_communicator.send_forward(
        output_tensor,
        send_tensor_shapes,
        is_pp_last_stage(p2p_communicator.pp_group)
    )

    if not forward_only:
        # Save activations for backward pass
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # Deallocate output tensor's data to save memory
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

**Key optimization**: `deallocate_output_tensor` sets `output_tensor.data = torch.empty((1,))` to free activation memory while keeping `.grad_fn` for backward pass.

**Steady-state phase** (1F1B):

```python
for i in range(num_microbatches_remaining):
    # Current microbatch index
    forward_k = i + num_warmup_microbatches

    # === 1 Forward pass ===
    input_tensor = p2p_communicator.recv_forward(...)
    output_tensor, num_tokens = forward_step(
        ...,
        input_tensor,
        ...,
        current_microbatch=forward_k,
    )
    p2p_communicator.send_forward(output_tensor, ...)

    # Save for backward
    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)
    deallocate_output_tensor(output_tensor, ...)

    # === 1 Backward pass ===
    # Process oldest microbatch (FIFO)
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    output_tensor_grad = p2p_communicator.recv_backward(...)
    input_tensor_grad = backward_step(
        input_tensor,
        output_tensor,
        output_tensor_grad
    )
    p2p_communicator.send_backward(input_tensor_grad, ...)
```

**Why FIFO?** Oldest microbatch has been waiting longest, reducing peak memory by freeing activations ASAP.

**Cooldown phase** (backward-only):

```python
for i in range(num_warmup_microbatches):
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    output_tensor_grad = p2p_communicator.recv_backward(...)
    input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)
    p2p_communicator.send_backward(input_tensor_grad, ...)

# After cooldown: input_tensors and output_tensors are empty (all processed)
```

### Interleaved 1F1B Implementation

**From `schedules.py:1059-1070`**, model chunk helpers:

```python
def get_model_chunk_id(virtual_microbatch_id, forward):
    """Get model chunk ID for this virtual microbatch."""
    model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
    if not forward:
        # Backward processes chunks in reverse order
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id

def get_microbatch_id_in_model_chunk(iteration_id, forward):
    """Get microbatch ID within the model chunk."""
    assert forward
    return microbatch_id_table[iteration_id]
```

**Schedule table construction** (from `schedules.py:1045-1057`):

```python
schedule_table = get_schedule_table(
    num_microbatches,
    len(model),  # num_model_chunks
    config.microbatch_group_size_per_vp_stage
)

# Example output for PP=2, M=5, VP=2, N=3:
# [(mb_id, chunk_id) for each virtual_mb_id]
# [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,0), (4,0), (3,1), (4,1)]
#   ^^^^^^^^^^^^^^^^^^^^^  Process N=3 mbs on each chunk before switching
```

**Main execution loop**:

```python
# Warmup phase
for virtual_microbatch_id in range(num_warmup_microbatches):
    model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
    microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

    # Use the appropriate model chunk
    current_model = model[model_chunk_id]
    current_data_iterator = data_iterator[model_chunk_id]

    # Forward on this specific model chunk
    output_tensor = forward_step(
        ...,
        current_model,
        ...,
        model_chunk_id=model_chunk_id,
        ...
    )

    # Save to chunk-specific list
    input_tensors[model_chunk_id].append(input_tensor)
    output_tensors[model_chunk_id].append(output_tensor)

# Steady-state: 1F1B with chunk switching
for virtual_microbatch_id in range(num_warmup_microbatches, total_num_microbatches):
    # Forward on next chunk
    forward_model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
    ... forward_step on model[forward_model_chunk_id] ...

    # Backward on oldest chunk
    backward_virtual_id = virtual_microbatch_id - num_warmup_microbatches
    backward_model_chunk_id = get_model_chunk_id(backward_virtual_id, forward=False)
    ... backward_step on model[backward_model_chunk_id] ...
```

**Key insight**: `model` is a list of model chunks, and we index into it based on the schedule table.

### Memory Management

**Activation deallocation** (from `schedules.py:135-146`):

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """Pseudo-deallocate output tensor's .data field."""
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor)
    assert out._base is None, "counter-productive to free a view of another tensor."

    # Set .data to scalar tensor (frees memory)
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)

    # .grad_fn is preserved for backward pass!
```

**Why this works**:
- Forward pass saves `output_tensor` with `.grad_fn` attached
- `.data` is freed immediately (large activation tensors)
- Backward pass only needs `.grad_fn` to recompute activations (if using activation checkpointing)
- Saves memory proportional to number of in-flight microbatches

**Peak memory calculation**:

```python
# Without deallocation
peak_memory = num_warmup_microbatches × activation_size + model_size

# With deallocation
peak_memory = 1 × activation_size + model_size  # Only need 1 activation at a time
# (assuming activation checkpointing recomputes during backward)
```

---

## Performance Optimizations

### 1. Overlapping P2P Communication

**Configuration**:
```bash
--overlap-p2p-comm  # Enable P2P overlap
```

**How it works**:
- Launch communication asynchronously
- Compute next microbatch while communication proceeds
- See [Communication Overlap](02-communication-overlap.md#pipeline-parallelism-overlap)

**Speedup**: 10-15% reduction in bubble time.

### 2. Batched P2P Operations

**Configuration**:
```bash
--batch-p2p-comm  # Enable batched P2P
```

**How it works**:
- Group multiple send/recv operations
- Launch all in a single `batch_isend_irecv`
- Reduces kernel launch overhead

**Speedup**: 5-10% for interleaved schedules with many virtual microbatches.

### 3. Activation Checkpointing

**Configuration**:
```bash
--recompute-granularity selective  # Checkpoint only expensive layers
--recompute-method block           # Checkpoint strategy
--num-microbatches-with-partial-activation-checkpoints 4  # Partial checkpointing
```

**Trade-off**:
- **Memory savings**: ~3-5× reduction in activation memory
- **Compute overhead**: ~20-30% more FLOPs (recomputing during backward)
- **Net benefit**: Enables larger models or more microbatches

### 4. Gradient Accumulation Fusion

**During cooldown**, gradients are accumulated across microbatches:

```python
# Naïve approach
for microbatch in microbatches:
    grad = backward(microbatch)
    param.grad += grad  # Accumulate

# Optimized approach
# Let PyTorch accumulate automatically (no explicit +=)
# Fuse gradient accumulation with optimizer step
```

**Speedup**: 10-15% for large models (fewer memory writes).

### 5. CUDA Graphs

**Configuration**:
```bash
--cuda-graph-mode full  # Capture entire schedule as CUDA graph
```

**How it works**:
- First iteration captures GPU operations
- Subsequent iterations replay the graph
- Eliminates Python overhead

**Speedup**: 5-10% for small models (Python overhead is significant).

**Limitations**:
- Requires fixed tensor shapes
- Not compatible with dynamic control flow

---

## Configuration Guide

### Basic Pipeline Configuration

```bash
# Standard 1F1B
python pretrain_gpt.py \
  --pipeline-model-parallel-size 4 \
  --num-layers 48 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --micro-batch-size 2 \
  --global-batch-size 64 \
  # global-batch-size = num-micro-batches × micro-batch-size
  # → num-micro-batches = 64 / 2 = 32
```

### Interleaved Pipeline Configuration

```bash
# Interleaved 1F1B with VP=2
python pretrain_gpt.py \
  --pipeline-model-parallel-size 8 \
  --virtual-pipeline-model-parallel-size 2 \
  --num-layers 96 \
  # Each GPU gets 96 / 8 = 12 layers, split into 2 chunks of 6 layers
  --micro-batch-size 1 \
  --global-batch-size 128 \
  # num-micro-batches = 128 / 1 = 128
  # Effective: 128 × 2 = 256 virtual microbatches
```

### Bubble Optimization

```bash
# High bubble (37.5%)
--pipeline-model-parallel-size 8 \
--global-batch-size 64 \
--micro-batch-size 2
# num-micro-batches = 32
# Bubble = (8-1)/32 = 21.9%

# Reduced bubble (9.4%)
--pipeline-model-parallel-size 8 \
--global-batch-size 128 \
--micro-batch-size 1
# num-micro-batches = 128
# Bubble = (8-1)/128 = 5.5%

# Virtual pipeline (4.7%)
--pipeline-model-parallel-size 8 \
--virtual-pipeline-model-parallel-size 2 \
--global-batch-size 64 \
--micro-batch-size 1
# num-micro-batches = 64, VP=2
# Bubble ≈ (8-1)/(64×2) = 5.5%
```

### Multi-Dimensional Parallelism

```bash
# TP + PP + DP
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 4 \
--global-batch-size 128 \
--micro-batch-size 2
# Total GPUs = 2 × 4 = 8
# DP size = 8 / (2×4) = 1 (no DP)
# num-micro-batches = 128 / 2 = 64

# To enable DP=2, need 16 GPUs total
# 16 = TP(2) × PP(4) × DP(2)
```

---

## Summary

### Key Takeaways

1. **1F1B schedule** reduces bubble time and memory compared to GPipe
2. **Warmup phase** fills pipeline, steady-state maintains constant memory
3. **Interleaved pipeline** (VP) cuts bubbles by ~VP× at cost of more memory
4. **Microbatch selection** critical: M ≥ 8 × PP for good efficiency
5. **P2P communication** uses batched isend/irecv for overlap
6. **Memory management** via activation deallocation and checkpointing

### Performance Targets

| Configuration | Bubble Time | Memory | Use Case |
|--------------|-------------|--------|----------|
| PP=4, M=32 | ~9% | Low | Standard large models |
| PP=8, M=64 | ~11% | Low | Very large models |
| PP=8, VP=2, M=64 | ~5% | Medium | Bubble-critical workloads |
| PP=16, VP=4, M=128 | ~3% | High | Extreme scale (1000+ GPUs) |

### Configuration Decision Matrix

```
if model fits on 1 GPU:
    PP = 1 (no pipeline)
elif bubble_critical and have_memory:
    use VP > 1 (interleaved)
elif memory_critical:
    use standard 1F1B + activation checkpointing
else:
    use standard 1F1B with M = 8 × PP
```

---

**References**:
- Megatron-LM Paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- 1F1B Schedule: [PipeDream Paper](https://arxiv.org/abs/1806.03377)
- Interleaved Pipeline: [Megatron-LM Code](https://github.com/NVIDIA/Megatron-LM)
- Memory Optimization: [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)
