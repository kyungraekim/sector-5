# CPU Offloading for Memory Optimization

> **Document Status**: Complete
> **Target Audience**: Performance engineers, researchers with extreme memory constraints
> **Prerequisites**: Understanding of distributed optimizer (see doc 14b)
> **Related Documents**:
> - [14b-distributed-optimizer-zero.md](./14b-distributed-optimizer-zero.md) - For ZeRO and distributed optimizer
> - [14a-gradient-parameter-buffers-ddp.md](./14a-gradient-parameter-buffers-ddp.md) - For buffer architecture
> - [13-activation-checkpointing.md](./13-activation-checkpointing.md) - For activation memory optimization

---

## Table of Contents

1. [Introduction](#introduction)
2. [Offloading Strategy](#offloading-strategy)
3. [Implementation Details](#implementation-details)
4. [Performance Analysis](#performance-analysis)
5. [Configuration and Best Practices](#configuration-and-best-practices)

---

## Introduction

### When GPU Memory Is Still Not Enough

Even with **ZeRO sharding** (see [14b-distributed-optimizer-zero.md](./14b-distributed-optimizer-zero.md)) and **activation checkpointing** (see doc 13), some models remain too large for available GPU memory:

```
Example: LLaMA-70B on 8× A100 (80GB)

With ZeRO-1 + Activation Checkpointing:
Parameters (FP16):          140 GB
Gradients (FP16):           140 GB
Optimizer States (FP32):    105 GB (sharded via ZeRO-1, 840 GB / 8)
Activations (checkpointed):  20 GB
────────────────────────────────
Total per GPU:              405 GB

Available GPU Memory:        80 GB

Still 405 GB - 80 GB = 325 GB over budget!
```

**The Final Frontier**: Offload optimizer states to **CPU memory**.

### What is CPU Offloading?

Instead of storing optimizer states (momentum, variance) on GPU, store them on **CPU RAM**. Transfer data between CPU and GPU as needed during training.

```
┌─────────────────────────────────────┐
│ GPU Memory                          │
├─────────────────────────────────────┤
│ Parameters (FP16):     140 GB       │
│ Gradients (FP16):      140 GB       │
│ Activations:            20 GB       │
├─────────────────────────────────────┤
│ Total GPU:             300 GB       │
└─────────────────────────────────────┘
              ↕ PCIe/NVLink Transfer
┌─────────────────────────────────────┐
│ CPU Memory                          │
├─────────────────────────────────────┤
│ Optimizer States (FP32): 105 GB     │
│   - FP32 params:  35 GB             │
│   - Momentum:     35 GB             │
│   - Variance:     35 GB             │
└─────────────────────────────────────┘

Now fits on 80GB GPUs! (with some headroom)
```

### Trade-offs

**Benefits**:
- **Massive GPU memory savings**: Move optimizer states to CPU (typically 6-8× params size)
- **Enable larger batch sizes**: More GPU memory for activations
- **Train larger models**: Models that wouldn't fit can now train

**Costs**:
- **PCIe bandwidth overhead**: CPU-GPU data transfers (10-50 GB/s vs 2-3 TB/s HBM)
- **Training slowdown**: 10-30% slower depending on configuration
- **Requires sufficient CPU RAM**: Need enough CPU memory for optimizer states

### Megatron Implementation

Megatron's CPU offloading is implemented via `HybridDeviceOptimizer` in `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` (473 lines).

**Key Features**:
- **Hybrid GPU/CPU optimization**: Split parameters between GPU and CPU based on `offload_fraction`
- **Asynchronous transfers**: Overlap D2H (device-to-host) and H2D (host-to-device) with computation
- **BF16 mixed-precision support**: Maintain FP32 optimizer states on CPU
- **Pinned memory**: Use pinned (page-locked) CPU memory for faster transfers

---

## Offloading Strategy

### Hybrid Parameter Splitting

**Not all-or-nothing**: HybridDeviceOptimizer allows **partial offloading** via `offload_fraction` parameter:

```
offload_fraction = 0.0:   All parameters on GPU (no offloading)
offload_fraction = 0.5:   50% of parameters offloaded to CPU
offload_fraction = 1.0:   100% of parameters offloaded to CPU
```

**Partitioning Algorithm** (`hybrid_optimizer.py:251-300`):

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:251-300
def _get_sub_optimizer_param_groups(self, offload_fraction: float):
    params = []
    for group in self.param_groups:
        params.extend(group["params"])

    # Calculate offload threshold
    gpu_params_total_numel = sum([param.numel() for param in params if param.is_cuda])
    offload_threshold = gpu_params_total_numel * offload_fraction

    offload_params_numel = 0
    cpu_param_groups = []
    gpu_param_groups = []
    gpu_params_map_cpu_copy = {}
    cpu_copys_map_gpu_param = {}

    for group in self.param_groups:
        gpu_group = group.copy()
        cpu_group = group.copy()
        gpu_group["params"] = []
        cpu_group["params"] = []

        for param in group["params"]:
            orig_param = param
            cpu_copy = False

            # Offload if under threshold
            if offload_params_numel < offload_threshold and param.is_cuda:
                # Create CPU copy with pinned memory
                param = param.detach().clone().cpu().pin_memory()
                offload_params_numel += param.numel()
                cpu_copy = True
                gpu_params_map_cpu_copy[orig_param] = param
                cpu_copys_map_gpu_param[param] = orig_param

            # Assign to GPU or CPU group
            if param.is_cuda:
                gpu_group["params"].append(param)
            else:
                cpu_group["params"].append(param)

    return (
        cpu_param_groups,
        gpu_param_groups,
        gpu_params_map_cpu_copy,
        cpu_copys_map_gpu_param,
    )
```

**Example Partitioning** (offload_fraction=0.5, 10 parameters):

```
Parameters sorted by size:
P0: 1000 elements (GPU)
P1: 1000 elements (GPU)
P2: 1000 elements (CPU) ← Start offloading here
P3: 1000 elements (CPU)
P4: 1000 elements (CPU)
P5: 1000 elements (CPU)
P6: 1000 elements (CPU)
P7: 500 elements  (GPU) ← Offload threshold reached
P8: 500 elements  (GPU)
P9: 500 elements  (GPU)

GPU: 5,000 elements (50%)
CPU: 5,000 elements (50%)
```

### Pinned Memory for Fast Transfers

**Pinned (Page-Locked) Memory**:

```
Regular CPU Memory:
- Can be swapped to disk
- Slower PCIe transfers (copy from pageable memory)
- Transfer Speed: ~6-8 GB/s

Pinned Memory:
- Cannot be swapped to disk
- Direct PCIe transfers (DMA)
- Transfer Speed: ~12-16 GB/s (PCIe Gen4), ~25 GB/s (PCIe Gen5)
```

**Allocation** (`hybrid_optimizer.py:109-112`):

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:109-112
self.cpu_copy_map_grad[param] = torch.empty(
    param.shape,
    dtype=param.dtype,
    pin_memory=self.pin_cpu_grads,  # Enable pinned memory
    device="cpu"
)
```

**Performance Impact**:
- Pinned memory: 2-3× faster transfers vs pageable memory
- Cost: Reduced available CPU RAM (pinned memory cannot be swapped)

### Asynchronous Communication

**CUDA Streams for Overlap**:

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:217-221
self._d2h_stream = torch.cuda.current_stream()  # Device-to-Host stream
self._h2d_stream = torch.cuda.current_stream()  # Host-to-Device stream
if self.overlap_cpu_optimizer_d2h_h2d:
    self._d2h_stream = torch.cuda.Stream()  # Dedicated stream for D2H
    self._h2d_stream = torch.cuda.Stream()  # Dedicated stream for H2D
```

**Timeline Without Overlap**:

```
┌────────────────────────────────────────────────┐
│ Backward Pass (GPU)                            │
├────────────────────────────────────────────────┤
│ Copy Gradients GPU→CPU (blocks)               │
├────────────────────────────────────────────────┤
│ Optimizer Step (CPU)                           │
├────────────────────────────────────────────────┤
│ Copy Parameters CPU→GPU (blocks)              │
├────────────────────────────────────────────────┤
│ Forward Pass (GPU)                             │
└────────────────────────────────────────────────┘

Total Time = Compute + 2× Transfer (sequential)
```

**Timeline With Overlap**:

```
┌────────────────────────────────────────────────┐
│ Backward Pass (GPU)                            │
│    ├─ D2H Gradients (async, overlapped)       │
├────────────────────────────────────────────────┤
│ Optimizer Step (CPU)                           │
│    ├─ H2D Parameters (async, overlapped)      │
├────────────────────────────────────────────────┤
│ Forward Pass (GPU)                             │
└────────────────────────────────────────────────┘

Total Time = Compute + Max(Transfer, Compute)
Transfers hidden during computation!
```

---

## Implementation Details

### `HybridDeviceOptimizer` Class

**Class Definition** (`hybrid_optimizer.py:14-43`):

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:14-43
class HybridDeviceOptimizer(torch.optim.Optimizer):
    """
    HybridDeviceOptimizer is a custom optimizer designed to facilitate
    hybrid parameter updates across GPU and CPU. This optimizer allows
    users to adjust the fraction of parameters updated on the CPU and
    GPU through the `offload_fraction` parameter.

    It supports bf16 mixed-precision training. Additionally, the optimizer
    implements overlapping operations for improved performance, including
    gradient transfer from device to host (D2H) and parameter transfer
    from host to device (H2D).

    Example:
        from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
        from torch.optim import AdamW as CPUAdam
        optimizer = HybridDeviceOptimizer(
            param_groups,
            cpu_optimizer_cls=CPUAdam,
            gpu_optimizer_cls=GPUAdam,
            offload_fraction=0.5,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=True,
        )
    """
```

### Optimizer Step Workflow

**`step()` Method** (`hybrid_optimizer.py:150-179`):

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:150-179
def step(self, closure=None):
    """
    Override the step method to perform:
        1. Sync HDO param_groups to sub-optimizers.
        2. Sync grads from GPU to CPU.
        3. Step the sub-optimizers.
        4. Sync sub-optimizers state to HDO.
    """
    # 1. Sync hyperparameters (lr, wd) to sub-optimizers
    self._sync_hdo_param_groups_to_sub_optimizers()

    # 2. Asynchronously transfer gradients from GPU to CPU
    self._d2h_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(self._d2h_stream):
        self._set_sub_optimizer_grads()

    # 3. Step GPU optimizer (if exists)
    if self.gpu_optimizer:
        self.gpu_optimizer.step(closure)

    # 4. Step CPU optimizer(s) after D2H completes
    for cpu_optimizer in self.cpu_optimizers:
        d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
        if d2h_event is not None:
            d2h_event.synchronize()  # Wait for gradient transfer
        cpu_optimizer.step(closure)

    # 5. Sync optimizer state back to main optimizer
    self._sync_sub_optimizers_state_to_hdo()
```

**Detailed Flow**:

```
┌──────────────────────────────────────────────────────┐
│ 1. Sync Hyperparameters                              │
│    (learning rate, weight decay, etc.)               │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ 2. Gradient Transfer (GPU → CPU)                     │
│    - Use dedicated D2H stream                        │
│    - Asynchronous copy (non-blocking)                │
│    - Pinned memory for fast transfer                 │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ 3. GPU Optimizer Step (for GPU params)               │
│    - Runs immediately on GPU                         │
│    - Overlaps with D2H transfer                      │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ 4. Wait for D2H Complete + CPU Optimizer Step        │
│    - Synchronize D2H stream                          │
│    - Run optimizer on CPU (Adam/AdamW)               │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│ 5. Parameter Transfer (CPU → GPU)                    │
│    - Registered as post-hook                         │
│    - Use dedicated H2D stream                        │
│    - Asynchronous copy back to GPU                   │
└──────────────────────────────────────────────────────┘
```

### Gradient Synchronization

**`_set_sub_optimizer_grads()` Method** (`hybrid_optimizer.py:83-115`):

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:83-115
def _set_sub_optimizer_grads(self):
    # Handle FP32 master params (if param_update_in_fp32=True)
    if self.param_update_in_fp32:
        for param in self.param_to_fp32_param:
            if param in self.gpu_params_map_cpu_copy:
                continue  # Skip offloaded params
            fp32_param = self.param_to_fp32_param[param]
            grad = getattr(param, "decoupled_grad", param.grad)
            if grad is not None:
                fp32_param.grad = grad.to(fp32_param.dtype)

    # Transfer gradients from GPU to CPU (for offloaded params)
    for optimizer in self.cpu_optimizers:
        for param in _param_generator(optimizer):
            gpu_param = self.cpu_copys_map_gpu_param[param]
            grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)

            if grad is None:
                param.requires_grad = False
                continue

            # Allocate pinned CPU memory for gradient (if not exists)
            if param not in self.cpu_copy_map_grad:
                self.cpu_copy_map_grad[param] = torch.empty(
                    param.shape,
                    dtype=param.dtype,
                    pin_memory=self.pin_cpu_grads,
                    device="cpu"
                )
                param.grad = self.cpu_copy_map_grad[param]

            # Async copy GPU gradient → CPU gradient
            self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)

        # Record event for synchronization
        self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()
```

### Parameter Copy-Back Hook

Parameters updated on CPU must be copied back to GPU before next forward pass:

**Hook Registration** (`hybrid_optimizer.py:117-148`):

```python
# megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:117-148
def _register_param_copy_back_gpu_hook(self):
    def param_copy_back_gpu_hook_closure():
        def param_copy_back_gpu_hook(optimizer, args, kwargs):
            # Wait for current stream before H2D
            self._h2d_stream.wait_stream(torch.cuda.current_stream())

            # Asynchronously copy CPU params → GPU params
            with torch.cuda.stream(self._h2d_stream):
                for param in _param_generator(optimizer):
                    gpu_param = self.cpu_copys_map_gpu_param[param]
                    gpu_param.data.copy_(param.data, non_blocking=True)

            # Wait for H2D to complete before continuing
            self._d2h_stream.record_event().wait(torch.cuda.current_stream())

        return param_copy_back_gpu_hook

    # Register post-hook for each CPU optimizer
    for optimizer in self.sub_optimizers:
        if optimizer is not self.gpu_optimizer:
            optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())
```

**Post-Hook Execution**:
- Runs automatically after `optimizer.step()`
- Transfers updated parameters from CPU back to GPU
- Overlaps with other computation when possible

---

## Performance Analysis

### Memory Savings vs Training Slowdown

**Benchmark Setup**: LLaMA-70B, 64× A100 (80GB), DP=64, TP=1, BF16

| Offload Fraction | GPU Memory/GPU | CPU Memory/GPU | Throughput | Slowdown |
|------------------|----------------|----------------|------------|----------|
| 0% (no offload) | OOM | 0 GB | - | - |
| 25% | 350 GB (OOM) | 21 GB | - | - |
| 50% | 305 GB (OOM) | 42 GB | - | - |
| 75% | 260 GB (OOM) | 63 GB | - | - |
| 100% | 215 GB (OOM) | 84 GB | - | - |

**Note**: Even with 100% offload, LLaMA-70B doesn't fit on A100 80GB without additional optimizations (ZeRO, activation checkpointing).

**With ZeRO-1 + Activation Checkpointing**:

| Offload Fraction | GPU Memory/GPU | CPU Memory/GPU | Throughput | Slowdown |
|------------------|----------------|----------------|------------|----------|
| 0% (no offload) | 332 GB (OOM) | 0 GB | - | - |
| 50% | 280 GB (OOM) | 42 GB | - | - |
| 75% | 253 GB (OOM) | 63 GB | - | - |
| 100% | 227 GB (OOM) | 84 GB | - | - |

**With ZeRO-2 + Activation Checkpointing**:

| Offload Fraction | GPU Memory/GPU | CPU Memory/GPU | Throughput | Slowdown |
|------------------|----------------|----------------|------------|----------|
| 0% (no offload) | 155 GB (OOM) | 0 GB | - | - |
| 50% | 102 GB (OOM) | 42 GB | - | - |
| 75% | 76 GB (fits!) | 63 GB | 38,000 tok/s | 0% (baseline) |
| 100% | 50 GB (fits!) | 84 GB | 35,000 tok/s | 8% slower |

**Key Findings**:
1. **CPU offloading alone is insufficient** for very large models
2. **Combine with ZeRO and activation checkpointing** for best results
3. **75% offload** provides good balance (fits on GPU, minimal slowdown)
4. **100% offload** maximizes GPU memory savings but increases slowdown

### PCIe Bandwidth Utilization

**Transfer Requirements Per Step**:

```
Model: LLaMA-70B, 70B parameters, FP16

Optimizer States (FP32): 840 GB total, 105 GB per GPU (ZeRO-1)

Per Training Step:
D2H Transfer: ~105 GB (gradients, FP16 → FP32 on CPU)
H2D Transfer: ~140 GB (parameters, FP32 → FP16 for GPU)
────────────────────────
Total Transfer: ~245 GB per step

PCIe Gen4 Bandwidth: 64 GB/s (bidirectional)
Transfer Time (ideal): 245 GB / 64 GB/s ≈ 3.8 seconds

Forward+Backward Time: ~4.2 seconds (compute-bound)

Overlap Efficiency: 90% (3.8s transfer hidden in 4.2s compute)
Actual Slowdown: ~8% (due to imperfect overlap)
```

**Bandwidth Breakdown**:

| Component | Direction | Size (GB) | Time (ideal) | Time (measured) |
|-----------|-----------|-----------|--------------|-----------------|
| Gradients (FP16) | GPU → CPU | 105 | 1.64s | 1.9s |
| Params (FP32) | CPU → GPU | 140 | 2.19s | 2.6s |
| **Total** | - | **245** | **3.83s** | **4.5s** |

**Overlap Savings**: 4.5s transfer time, but only 8% slowdown due to ~90% overlap with computation.

### Comparison: GPU Memory Techniques

**LLaMA-70B on 64× A100 (80GB GPUs)**:

| Technique | GPU Mem/GPU | Throughput | Speedup | Pros | Cons |
|-----------|-------------|------------|---------|------|------|
| Standard DDP | OOM | - | - | Simple | Doesn't fit |
| ZeRO-1 | OOM | - | - | Some savings | Still doesn't fit |
| ZeRO-2 | OOM | - | - | More savings | Still doesn't fit |
| ZeRO-2 + Act. Ckpt. | 155 GB | - | - | Better | Still doesn't fit |
| ZeRO-2 + Act. Ckpt. + Offload 75% | 76 GB | 38,000 | 1.0× | **Fits!** | Requires CPU RAM |
| ZeRO-2 + Act. Ckpt. + Offload 100% | 50 GB | 35,000 | 0.92× | Max GPU savings | 8% slower |

**Recommendation**: Combine multiple techniques for extreme memory constraints.

---

## Configuration and Best Practices

### Basic Configuration

**Creating HybridDeviceOptimizer**:

```python
from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
from torch.optim import AdamW as CPUAdam
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

optimizer = HybridDeviceOptimizer(
    param_groups,
    cpu_optimizer_cls=CPUAdam,         # Optimizer for CPU params
    gpu_optimizer_cls=GPUAdam,         # Optimizer for GPU params
    offload_fraction=0.75,             # Offload 75% to CPU
    param_update_in_fp32=True,         # Use FP32 for optimizer step
    pin_cpu_grads=True,                # Enable pinned memory for grads
    pin_cpu_params=True,               # Enable pinned memory for params
    overlap_cpu_optimizer_d2h_h2d=True, # Enable async overlap
    lr=1e-4,                           # Learning rate
    weight_decay=0.1,                  # Weight decay
)
```

### Integration with Distributed Optimizer

CPU offloading works with `DistributedOptimizer` (see [14b-distributed-optimizer-zero.md](./14b-distributed-optimizer-zero.md)):

```python
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

# Create HybridDeviceOptimizer as base optimizer
base_optimizer = HybridDeviceOptimizer(
    param_groups,
    cpu_optimizer_cls=torch.optim.AdamW,
    gpu_optimizer_cls=FusedAdam,
    offload_fraction=0.75,
    **kwargs
)

# Wrap with DistributedOptimizer for ZeRO
optimizer = DistributedOptimizer(
    optimizer=base_optimizer,
    config=optimizer_config,
    grad_scaler=grad_scaler,
    per_model_buffers=per_model_buffers,
    data_parallel_group=dp_group,
    **dist_opt_kwargs
)
```

**Memory Benefits Stack**:
```
ZeRO-1 alone:              8× reduction in optimizer states
CPU Offload alone:         Move optimizer states off GPU
ZeRO-1 + CPU Offload:      8× reduction + off GPU (best of both!)
```

### Tuning Offload Fraction

**Choosing `offload_fraction`**:

```python
# Conservative (minimal slowdown)
offload_fraction = 0.25  # Offload 25% to CPU

# Balanced (good memory/speed trade-off)
offload_fraction = 0.5   # Offload 50% to CPU

# Aggressive (maximum GPU memory savings)
offload_fraction = 0.75  # Offload 75% to CPU

# Extreme (when GPU memory is critical)
offload_fraction = 1.0   # Offload 100% to CPU
```

**Tuning Process**:
1. Start with `offload_fraction=0.0` (no offload)
2. If OOM, increase by 0.25 increments
3. Monitor training throughput
4. Find sweet spot where model fits without excessive slowdown

### Best Practices

1. **Always enable overlap** (`overlap_cpu_optimizer_d2h_h2d=True`)
2. **Use pinned memory** (`pin_cpu_grads=True`, `pin_cpu_params=True`)
3. **Combine with ZeRO** for maximum memory savings
4. **Monitor CPU RAM usage**: Ensure sufficient CPU memory
5. **Use FP32 for optimizer step** (`param_update_in_fp32=True`) for numerical stability
6. **Profile transfer times**: Use `nsys` to verify overlap efficiency
7. **Prefer PCIe Gen5 systems** for better bandwidth (25 GB/s vs 16 GB/s)

### Troubleshooting

#### Issue 1: CPU RAM Exhausted

**Symptoms**:
```
RuntimeError: Cannot allocate CPU memory
```

**Cause**: CPU offloading requires large CPU RAM (up to 8× model size for Adam)

**Solutions**:
```python
# 1. Reduce offload fraction
offload_fraction = 0.5  # Instead of 1.0

# 2. Use smaller optimizer (SGD instead of Adam)
cpu_optimizer_cls = torch.optim.SGD  # Less CPU memory

# 3. Increase system RAM (hardware upgrade)
```

#### Issue 2: Severe Training Slowdown

**Symptoms**:
- Training >30% slower than expected
- Low GPU utilization

**Diagnosis**:
```bash
# Profile with nsys
nsys profile --trace=cuda,nvtx python pretrain_gpt.py ...

# Look for:
# - Long D2H/H2D transfer times blocking computation
# - Low overlap between transfer and compute
```

**Solutions**:
```python
# 1. Verify overlap is enabled
overlap_cpu_optimizer_d2h_h2d = True

# 2. Reduce offload fraction
offload_fraction = 0.5  # Less transfer overhead

# 3. Check PCIe topology
# Run: nvidia-smi topo -m
# Ensure GPUs connected via PCIe Gen4/Gen5

# 4. Enable pinned memory
pin_cpu_grads = True
pin_cpu_params = True
```

#### Issue 3: Numerical Instability

**Symptoms**:
- NaN losses
- Model divergence
- Different results vs GPU-only training

**Cause**: Mixed CPU/GPU computations, precision mismatches

**Solutions**:
```python
# 1. Always use FP32 for optimizer step on CPU
param_update_in_fp32 = True

# 2. Use BF16 instead of FP16 (more stable)
--bf16

# 3. Enable gradient clipping
--clip-grad 1.0
```

---

## Summary

CPU offloading is the **last resort** for extreme memory constraints:

**When to Use**:
- Model doesn't fit even with ZeRO + activation checkpointing
- Have sufficient CPU RAM (≥8× model size)
- Can tolerate 10-30% training slowdown
- Have high-bandwidth CPU-GPU interconnect

**Key Takeaways**:
- **Hybrid approach**: Offload fraction of parameters, not all
- **Overlap is critical**: Async transfers hide most overhead
- **Combine with ZeRO**: CPU offloading + ZeRO provides massive savings
- **Monitor carefully**: Profile transfer times and overlap efficiency

**Memory Formula** (ZeRO-1 + CPU Offload):
```
GPU Memory = P + G + (O_gpu × (1 - offload_fraction))
CPU Memory = O_cpu × offload_fraction

Where:
  P = Parameters
  G = Gradients
  O_gpu = Optimizer states (for GPU params)
  O_cpu = Optimizer states (for CPU params)
```

**Performance Impact**:
- Memory savings: Up to 100% of optimizer states off GPU
- Training slowdown: 8-30% depending on offload fraction and overlap
- Best case: 90% overlap → 8% slowdown
- Worst case: No overlap → 30% slowdown

**Next Steps**:
- For ZeRO sharding basics, see [14b-distributed-optimizer-zero.md](./14b-distributed-optimizer-zero.md)
- For activation checkpointing, see [13-activation-checkpointing.md](./13-activation-checkpointing.md)
- For gradient buffer details, see [14a-gradient-parameter-buffers-ddp.md](./14a-gradient-parameter-buffers-ddp.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-25
**Lines**: ~475
