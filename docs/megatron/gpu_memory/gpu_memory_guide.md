# GPU Memory Guide: Understanding, Estimating, and Preventing OOM in LLM Training

This guide explains where GPU memory goes during LLM training with Megatron-LM, how to
predict whether a given configuration will fit in memory, why OOM errors sometimes appear
after many training steps, and what to do about it.

---

## Table of Contents

1. [Where GPU Memory Goes](#where-gpu-memory-goes)
2. [Estimating Memory Before Training](#estimating-memory-before-training)
3. [Using Megatron's Built-In Memory Estimator](#using-megatrons-built-in-memory-estimator)
4. [How Parallelism Reduces Memory](#how-parallelism-reduces-memory)
5. [Why OOM Happens After Many Steps](#why-oom-happens-after-many-steps)
6. [Runtime Memory Monitoring](#runtime-memory-monitoring)
7. [Memory Reduction Strategies](#memory-reduction-strategies)
8. [Practical Workflow: Will My Training Fit?](#practical-workflow-will-my-training-fit)

---

## Where GPU Memory Goes

During mixed-precision training (BF16 params + FP32 optimizer), GPU memory is consumed by
four major categories:

### 1. Model States (Parameters + Optimizer + Gradients)

| Component | Per-Parameter Bytes | Description |
|-----------|--------------------:|-------------|
| BF16 parameters | 2 | The model weights used in forward/backward |
| FP32 optimizer master copy | 4 | Adam maintains an FP32 copy of each parameter |
| FP32 first moment (Adam) | 4 | Running mean of gradients |
| FP32 second moment (Adam) | 4 | Running mean of squared gradients |
| BF16 gradients | 2 | Accumulated during backward pass (FP32 → 4B if `--accumulate-allreduce-grads-in-fp32`) |
| **Total** | **~18** | **Per parameter, per GPU (without distributed optimizer)** |

**Example**: A 70B parameter model requires ~70B × 18 = **1,260 GB** of model state memory
across all GPUs (before any parallelism sharding).

### 2. Activation Memory

Activations are the intermediate tensors saved during the forward pass for use in the backward
pass. This is typically the **largest and most variable** memory consumer.

For a single transformer layer with sequence parallelism and selective recomputation
(`core_attn`), per the formula from
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
(Table 2):

```
activation_per_layer = s × b × h × (18 + 4 × ffn_hidden_size / hidden_size)
```

Where `s` = sequence length, `b` = micro-batch size, `h` = hidden size.

**Without sequence parallelism**, activations are larger:

```
activation_per_layer = s × b × h × (10 + 24 / tp)
```

Total activation memory scales with:
- **Number of layers per pipeline stage** (`num_layers / pp`)
- **Number of microbatches in flight** (pipeline bubble)
- **Interleaved schedule penalty** (if virtual pipeline parallelism is used)

**Example**: For Llama-70B (h=8192, ffn=28672, s=4096, b=1, 80 layers, PP=4, TP=8, SP on):

```
per_layer = 4096 × 1 × 8192 × (18 + 4 × 28672/8192) = 4096 × 1 × 8192 × 32 = 1.07 GB
per_stage = 1.07 GB × (80/4) = 21.5 GB
per_gpu   = 21.5 GB / 8 (TP) ≈ 2.7 GB
```

### 3. Temporary Buffers and Workspace

These are less predictable and include:

| Buffer | Typical Size | Notes |
|--------|-------------|-------|
| Communication buffers (all-gather, reduce-scatter) | 100 MB – 2 GB | Depends on TP/PP overlap settings |
| cuBLAS workspace | 100 – 500 MB | Per-GPU, allocated lazily by CUDA |
| NCCL internal buffers | 200 MB – 1 GB | Depends on number of process groups |
| PyTorch CUDA caching allocator overhead | 5 – 15% | Fragmentation causes reserved > allocated |
| Global memory buffers (`GlobalMemoryBuffer`) | Variable | Reused across iterations |

### 4. Other

| Item | Size | Notes |
|------|------|-------|
| CUDA context | 500 MB – 1.5 GB | Fixed overhead per GPU, depends on driver/GPU |
| Python/framework overhead | 200 – 500 MB | PyTorch, NCCL libraries loaded in GPU memory |
| Embedding tables | `vocab_size × h × 2B` | Can be large for big vocabularies |

### Memory Breakdown Visualization

For a typical 70B model training run (per GPU, with TP=8, PP=4, DP=4, distributed optimizer):

```
┌─────────────────────────────────────────────────┐
│                 80 GB GPU Memory                 │
├─────────────────────────────────────────────────┤
│ CUDA context + framework        │   ~1.5 GB     │
├─────────────────────────────────────────────────┤
│ Parameters (BF16)               │   ~4.4 GB     │ ← 70B / (TP×PP) × 2B
├─────────────────────────────────────────────────┤
│ Optimizer states                │   ~5.5 GB     │ ← Sharded across DP with dist optim
├─────────────────────────────────────────────────┤
│ Gradients                       │   ~4.4 GB     │
├─────────────────────────────────────────────────┤
│ Activations                     │   ~2.7 GB     │ ← With selective recompute + SP
├─────────────────────────────────────────────────┤
│ Communication buffers           │   ~1.0 GB     │
├─────────────────────────────────────────────────┤
│ Temporary / workspace           │   ~1.0 GB     │
├─────────────────────────────────────────────────┤
│ Fragmentation overhead          │   ~2.0 GB     │ ← Reserved but not allocated
├─────────────────────────────────────────────────┤
│ FREE                            │  ~57.5 GB     │
└─────────────────────────────────────────────────┘
```

---

## Estimating Memory Before Training

### Manual Calculation

#### Step 1: Count Parameters on the Most-Loaded GPU

The most-loaded shard (typically the first or last pipeline stage, which holds the embedding)
has:

```
params_on_gpu = (transformer_params / PP + embedding_params) / TP
```

For a standard transformer layer (non-MoE):

```
params_per_layer = 2 × h × (4h_ffn × gated_multiplier + 2) + self_attention_params

self_attention_params (standard) = 2 × h² × (1 + num_kv_heads / num_heads)
self_attention_params (GQA)      = h × (h + 2 × kv_channels × num_kv_heads + h)

embedding_params = vocab_size × h  (×2 if untied output weights)
```

#### Step 2: Compute Model State Memory

```python
# Without distributed optimizer
model_state_bytes = params_on_gpu * 18  # 2 + 4 + 4 + 4 + 4 bytes

# With distributed optimizer (--use-distributed-optimizer)
model_state_bytes = params_on_gpu * (6 + 12 / dp_size)  # BF16 params/grads + sharded optim
```

#### Step 3: Estimate Activation Memory

```python
# With sequence parallelism + selective recompute (core_attn)
activation_bytes = (
    seq_len * micro_batch * hidden * (18 + 4 * ffn_hidden / hidden)
    * (num_layers / pp_size)
    / tp_size
)

# Without sequence parallelism
activation_bytes = (
    seq_len * micro_batch * hidden * (10 + 24 / tp_size)
    * (num_layers / pp_size)
)
```

Multiply by the pipeline interleave penalty if using virtual pipeline parallelism:

```python
if virtual_pp_size is not None:
    penalty = 1 + (pp_size - 1) / (pp_size * virtual_pp_size)
    activation_bytes *= penalty
```

#### Step 4: Add Overhead

```python
overhead = 2.5e9  # ~2.5 GB for CUDA context, cuBLAS, NCCL, etc.
fragmentation_factor = 1.10  # 10% fragmentation overhead

total_estimated = (model_state_bytes + activation_bytes) * fragmentation_factor + overhead
```

#### Step 5: Compare to GPU Memory

```
If total_estimated < GPU_memory × 0.95:  → Likely fits
If total_estimated ≈ GPU_memory × 0.90-0.95:  → Tight, may OOM on peaks
If total_estimated > GPU_memory × 0.95:  → Will OOM
```

---

## Using Megatron's Built-In Memory Estimator

Megatron provides `tools/report_theoretical_memory.py` to compute theoretical memory
**without instantiating a model or running on GPUs**:

```bash
python tools/report_theoretical_memory.py \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --micro-batch-size 1 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --vocab-size 32000 \
    --padded-vocab-size 32000 \
    --sequence-parallel \
    --recompute-granularity selective \
    --use-distributed-optimizer \
    --bf16 \
    --verbose
```

This outputs:

```
Number of parameters in transformer block in billions: X.XX
Number of parameters in embedding layers in billions: X.XX
Total number of parameters in billions: X.XX
Number of parameters in most loaded shard in billions: X.XXXX
Activation memory footprint per transformer layer: XXX.X MB
Theoretical memory footprints: weight and optimizer=XXXX.XX MB, activation=XXXX.XX MB, total=XXXXX.XX MB
```

### What the Estimator Covers

The function `report_theoretical_memory()` in
`megatron/training/theoretical_memory_usage.py` computes:

1. **Weight + optimizer memory** (`compute_weight_and_optimizer_memory`):
   - Counts parameters per transformer layer (attention + MLP + layernorms)
   - Handles MoE layers (different parameter count per expert layer vs dense layer)
   - Handles MLA (Multi-Latent Attention) parameter structure
   - Handles MTP (Multi-Token Prediction) extra layers
   - Accounts for embedding sharing/untieing
   - Computes most-loaded shard: `(transformer_params / PP + mtp_params + embedding) / TP`
   - Applies bytes-per-parameter: `18` (standard) or `6 + 12/dp` (distributed optimizer)

2. **Activation memory** (`compute_activation_memory` / `compute_activation_memory_without_sp`):
   - Uses formulas from the selective recomputation paper
   - Accounts for pipeline schedule (interleaved vs non-interleaved)
   - Includes embedding + output layer activations
   - Divides by TP size for sequence parallelism

### What the Estimator Does NOT Cover

- Communication buffers (all-gather, reduce-scatter overlap)
- cuBLAS / NCCL workspace allocations
- PyTorch caching allocator fragmentation
- CUDA context overhead (~1-1.5 GB)
- Temporary tensors during loss computation (especially large-vocabulary softmax)
- Memory spikes from data-dependent operations (e.g., MoE token dispatching)

**Rule of thumb**: Add 15-20% to the estimator's output for a realistic peak memory estimate.

---

## How Parallelism Reduces Memory

Each parallelism strategy shards different memory components:

| Strategy | Parameters | Optimizer | Gradients | Activations | Communication Cost |
|----------|:---------:|:---------:|:---------:|:-----------:|:------------------:|
| **TP** (Tensor Parallel) | ÷ TP | ÷ TP | ÷ TP | ÷ TP (with SP) | All-reduce per layer |
| **PP** (Pipeline Parallel) | ÷ PP | ÷ PP | ÷ PP | ÷ PP (per stage) | P2P per micro-batch |
| **DP** (Data Parallel) | — | — | — | — | All-reduce per step |
| **DP + Distributed Optimizer** | — | ÷ DP | ÷ DP | — | Reduce-scatter + all-gather |
| **FSDP** (`optim_grads_params`) | ÷ DP | ÷ DP | ÷ DP | — | All-gather on forward + backward |
| **CP** (Context Parallel) | — | — | — | ÷ CP | Ring attention comm |
| **EP** (Expert Parallel) | ÷ EP (experts only) | ÷ EP | ÷ EP | Varies | All-to-all per MoE layer |
| **SP** (Sequence Parallel) | — | — | — | ÷ TP | Fused with TP comm |

**Total GPU count**: `TP × PP × CP × EP × DP`

### Memory Formula by Parallelism

```python
# Model states per GPU
params_per_gpu = total_params / (TP * PP)  # EP divides expert params only
if distributed_optimizer:
    optim_per_gpu = params_per_gpu * 12 / DP  # FP32 master + m1 + m2
    grad_per_gpu = params_per_gpu * 2          # BF16 grad (reduced via reduce-scatter)
else:
    optim_per_gpu = params_per_gpu * 12        # Replicated
    grad_per_gpu = params_per_gpu * 2          # Replicated (all-reduced)

# With FSDP (optim_grads_params): params also sharded
if fsdp:
    params_per_gpu = total_params / (TP * PP * DP)

# Activations per GPU
activation_per_gpu = activation_per_layer * (num_layers / PP) / TP  # SP divides by TP
if context_parallel:
    activation_per_gpu /= CP  # CP shards along sequence dimension
```

---

## Why OOM Happens After Many Steps

This is one of the most frustrating failure modes in large-scale training. The model fits
in memory for hundreds or thousands of steps, then suddenly OOMs. There are several causes:

### 1. PyTorch CUDA Caching Allocator Fragmentation

**The most common cause.** PyTorch's memory allocator uses a caching strategy — when a tensor
is freed, its memory block is returned to a cache, not to CUDA. Over many iterations:

- Blocks of different sizes accumulate in the cache
- The allocator cannot find a contiguous block large enough for a new allocation
- Even though total free memory is sufficient, the allocation fails

**Symptoms**: `torch.cuda.memory_allocated()` shows plenty of free memory, but allocation
still fails. The error message often says something like "tried to allocate X MB" where X
is much less than available memory.

**Fix**: Set the `PYTORCH_CUDA_ALLOC_CONF` environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`expandable_segments:True` (available since PyTorch 2.1) allows the allocator to create
segments that can grow, dramatically reducing fragmentation. This is the single most impactful
setting for preventing late-stage OOM.

Additional allocator tuning:

```bash
# Combine multiple settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# max_split_size_mb: Prevents the allocator from splitting blocks larger than this.
# Smaller values reduce fragmentation but may increase memory usage.
# Try 256, 512, or 1024 depending on your workload.
```

### 2. Variable-Length Input Data

If your data contains sequences of varying lengths (even with padding), some batches may
have more actual tokens or different padding patterns that change the memory footprint of:
- Attention score matrices (quadratic in actual sequence length for some kernels)
- MoE token routing (number of tokens per expert varies)
- Dynamic shapes in Flash Attention

**Fix**: Use consistent sequence lengths or set `--pad-to-max-length`. Monitor peak memory
across many batches during initial training steps.

### 3. Lazy CUDA Library Initialization

Some CUDA libraries allocate workspace memory on first use, not at initialization:

- **cuBLAS**: Allocates workspace on first GEMM call of each shape
- **cuDNN**: Allocates workspace on first convolution or attention call
- **NCCL**: May allocate additional buffers as communication patterns change

Over the first few hundred steps, as different code paths are exercised (warmup, main
training, evaluation), these libraries progressively claim more GPU memory.

**Fix**: Run several warmup steps and measure peak memory after the model has "settled."
Megatron typically sees peak memory within the first 5-10 steps, but evaluation/logging
steps may trigger new allocations.

### 4. Evaluation / Logging Steps

Periodic evaluation or logging can trigger additional memory usage:

- **Validation forward pass**: May use different batch sizes or sequence lengths
- **Loss computation over full vocabulary**: Softmax over `vocab_size` creates a large tensor
- **Metric computation**: Additional temporary tensors
- **TensorBoard/WandB logging**: Serialization buffers

**Fix**: Ensure your memory budget accounts for eval steps. If necessary, use
`torch.cuda.empty_cache()` before evaluation, or reduce eval batch size.

### 5. Gradient Accumulation Edge Cases

With gradient accumulation, some buffers may grow over accumulation steps:

- Gradient buffers for parameters with `reduce_scatter` timing
- MoE auxiliary loss accumulators
- Custom loss accumulators

**Fix**: Monitor memory across a full gradient accumulation cycle, not just one micro-batch.

### 6. Python/Framework Memory Leaks

Rare but possible:

- Tensors accidentally held by Python references (e.g., in logging dicts, metric accumulators)
- Growing lists of loss values or statistics
- Profiler or debugger hooks that accumulate history

**Fix**: Use `torch.cuda.memory._snapshot()` to dump detailed allocation history and analyze
with PyTorch's memory visualizer.

---

## Runtime Memory Monitoring

### Megatron Built-In Monitoring

**TensorBoard logging** (enable with CLI flag):

```bash
--log-memory-to-tensorboard
```

This logs the following metrics every iteration to TensorBoard:

| Metric | What it measures |
|--------|-----------------|
| `mem-reserved-bytes` | Total memory held by the PyTorch caching allocator |
| `mem-allocated-bytes` | Memory currently in use by tensors |
| `mem-max-allocated-bytes` | Peak allocated memory since last reset |
| `mem-allocated-count` | Number of active tensor allocations |

Look at these in TensorBoard to spot memory trends. A steadily increasing
`mem-reserved-bytes` without a corresponding increase in `mem-allocated-bytes` indicates
fragmentation.

**Memory snapshot** (for deep debugging):

```bash
--record-memory-history --memory-snapshot-path snapshot.pickle
```

This dumps a detailed memory history pickle on the last rank at every `--log-interval`.
Analyze it with PyTorch's memory visualization tools:

```python
import pickle
from torch.cuda._memory_viz import segment_plot

with open("snapshot.pickle", "rb") as f:
    snapshot = pickle.load(f)

# Generate an interactive HTML visualization
with open("memory_viz.html", "w") as f:
    f.write(segment_plot(snapshot))
```

### Manual Monitoring in Code

For custom monitoring or debugging:

```python
import torch

# Current and peak memory
allocated = torch.cuda.memory_allocated() / 1e9  # GB
reserved = torch.cuda.memory_reserved() / 1e9    # GB
peak = torch.cuda.max_memory_allocated() / 1e9    # GB

print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Peak: {peak:.2f} GB")

# Reset peak stats (useful for per-step measurement)
torch.cuda.reset_peak_memory_stats()

# Full memory stats dict
stats = torch.cuda.memory_stats()
print(f"Active allocations: {stats['allocation.all.current']}")
print(f"Inactive split blocks: {stats['inactive_split_bytes.all.current'] / 1e9:.2f} GB")
```

Key metrics to watch:

| Metric | Meaning | Concern Threshold |
|--------|---------|-------------------|
| `allocated_bytes.all.peak` | Peak memory actually used by tensors | Within 5% of GPU memory |
| `reserved_bytes.all.current` - `allocated_bytes.all.current` | Fragmentation waste | > 20% of reserved |
| `inactive_split_bytes.all.current` | Memory in split but unused blocks | Growing over time |

---

## Memory Reduction Strategies

Ordered by ease-of-use and impact. Try these in order:

### Tier 1: Easy, High Impact

| Strategy | CLI Flag | Memory Savings | Compute Cost |
|----------|----------|:-------------:|:------------:|
| Distributed optimizer | `--use-distributed-optimizer` | Optimizer: ÷ DP | ~0% (minor comm overhead) |
| Selective recomputation | `--recompute-granularity selective` | 30-60% of activations | < 5% |
| Sequence parallelism | `--sequence-parallel` | Activations: ÷ TP | ~0% (fused with TP) |
| Reduce micro-batch size | `--micro-batch-size N` (lower) | Linear in batch size | Reduces throughput |
| Allocator tuning | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Eliminates fragmentation | 0% |

### Tier 2: Moderate Effort

| Strategy | CLI Flag | Memory Savings | Compute Cost |
|----------|----------|:-------------:|:------------:|
| Full recomputation | `--recompute-granularity full` | Nearly all activations | 25-40% |
| More aggressive selective recompute | `--recompute-modules core_attn layernorm mlp` | Additional activations | 5-15% |
| Increase TP | `--tensor-model-parallel-size` (higher) | All components: ÷ TP | Higher comm overhead |
| Increase PP | `--pipeline-model-parallel-size` (higher) | All components: ÷ PP | Pipeline bubbles |
| Context parallelism | `--context-parallel-size N` | Activations for long seq | Ring attention comm |

### Tier 3: Advanced

| Strategy | How | Memory Savings | Trade-off |
|----------|-----|:-------------:|:---------:|
| FSDP | `--data-parallel-sharding-strategy optim_grads_params` | Params + optim + grads: ÷ DP | All-gather on every forward |
| FP8 training | `--fp8 e4m3` (requires TE + Hopper GPU) | ~50% param/activation memory | Minor accuracy impact |
| CPU offloading | `--cpu-offloading` | Offload activations/weights | CPU↔GPU transfer latency |
| Flash Attention | Enabled by default with TE | O(s) instead of O(s²) attention | None (free improvement) |

### Quick Decision Tree

```
OOM during training?
│
├─ Not using distributed optimizer?
│  └─ Add --use-distributed-optimizer              ← Try this first
│
├─ Not using selective recomputation?
│  └─ Add --recompute-granularity selective         ← Best memory/compute ratio
│
├─ Not using sequence parallelism?
│  └─ Add --sequence-parallel                       ← Free with TP
│
├─ OOM happens after many steps (not step 1)?
│  └─ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  ← Fragmentation fix
│
├─ Still OOM?
│  ├─ Reduce --micro-batch-size                     ← Direct activation reduction
│  ├─ Add more recompute modules (layernorm, mlp)   ← Progressive savings
│  └─ Increase TP or PP                             ← More parallelism
│
└─ Still OOM?
   ├─ --recompute-granularity full                  ← Nuclear option for activations
   ├─ FSDP with full sharding                       ← Nuclear option for model states
   └─ FP8 training                                  ← Halves many memory components
```

---

## Practical Workflow: Will My Training Fit?

Follow this step-by-step process before launching a long training run:

### Step 1: Analytical Estimate

Use Megatron's built-in estimator:

```bash
python tools/report_theoretical_memory.py \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --micro-batch-size 1 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --recompute-granularity selective \
    --bf16 \
    --verbose
```

Add 15-20% overhead to the reported total. If this exceeds 90% of GPU memory, adjust the
config before proceeding.

### Step 2: Short Dry Run (5-10 Steps)

Run your actual training config for a few steps:

```bash
# Add memory logging
--log-memory-to-tensorboard

# Set allocator config
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun ... your_training_script.py ...
```

After the run, check peak memory:

```python
# Add this after a few training steps
import torch
peak_gb = torch.cuda.max_memory_allocated() / 1e9
reserved_gb = torch.cuda.max_memory_reserved() / 1e9
gpu_total_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"Peak: {peak_gb:.2f} GB / {gpu_total_gb:.2f} GB ({peak_gb/gpu_total_gb*100:.1f}%)")
print(f"Fragmentation: {(reserved_gb - peak_gb):.2f} GB ({(reserved_gb-peak_gb)/reserved_gb*100:.1f}%)")
```

### Step 3: Safety Margin Check

| Peak Memory / GPU Total | Verdict | Action |
|:-----------------------:|---------|--------|
| < 80% | Safe | Proceed with training |
| 80 – 90% | Caution | Monitor closely; ensure `expandable_segments:True` |
| 90 – 95% | Risky | Likely to OOM on memory spikes; apply Tier 1 reductions |
| > 95% | Will fail | Must reduce memory usage before long training |

### Step 4: Monitor During Training

Enable TensorBoard memory logging (`--log-memory-to-tensorboard`) and watch for:

- **Steady increase** in `mem-reserved-bytes` → fragmentation (fix with allocator config)
- **Periodic spikes** in `mem-allocated-bytes` → eval steps or variable data (reduce eval
  batch or pad sequences)
- **Step function increase** in `mem-allocated-bytes` → new code path triggered (e.g., first
  eval step, learning rate warmup complete)

### Summary: Memory Estimation Accuracy

| Component | Estimable? | Accuracy |
|-----------|:----------:|:--------:|
| Parameters | Yes | ~100% |
| Optimizer states | Yes | ~100% |
| Gradients | Yes | ~100% |
| Activations (with SP + selective recompute) | Yes | ~90% |
| Activations (without SP, full recompute) | Partially | ~80% |
| Communication buffers | No | — |
| CUDA context + workspace | No | — |
| Fragmentation | No | — |
| Data-dependent spikes | No | — |

**Bottom line**: You can predict ~80-85% of total memory usage analytically. The remaining
15-20% from fragmentation, temporary buffers, and data-dependent peaks requires empirical
measurement. Always run a short dry run and maintain a 10-15% safety margin for production
training.

---

## See Also

- [`docs/selective_recomputation_guide.md`](selective_recomputation_guide.md) — Detailed guide
  on selective activation recomputation modules
- [`docs/selective_mlp_checkpointing_guide.md`](selective_mlp_checkpointing_guide.md) —
  Per-layer MLP checkpointing
- `megatron/training/theoretical_memory_usage.py` — Source code for the memory estimator
- `tools/report_theoretical_memory.py` — CLI tool for memory estimation
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) —
  Paper describing the selective recomputation approach
