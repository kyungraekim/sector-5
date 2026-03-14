# Transformer Engine Configuration Reference

> **Practical guide to configuring Transformer Engine for optimal performance**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [CLI Flags Reference](#cli-flags-reference)
4. [Configuration Recipes by Model Type](#configuration-recipes-by-model-type)
5. [Hardware-Specific Guides](#hardware-specific-guides)
6. [FP8 Configuration Deep Dive](#fp8-configuration-deep-dive)
7. [Communication Optimization](#communication-optimization)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Performance Tuning Checklist](#performance-tuning-checklist)

---

## Overview

NVIDIA Transformer Engine (TE) is the **single most impactful optimization** for Megatron-LM training, providing:

- **2× throughput** on Hopper GPUs (H100/H200) with FP8
- **1.2-1.4× speedup** on Ampere GPUs (A100) even without FP8
- **20-30% faster communication** with symmetric all-reduce
- **Fused operations** (LayerNorm+Linear, RoPE, MLP, cross-entropy)
- **Automatic mixed precision** for FP8 with minimal accuracy loss

### When to Use Transformer Engine

**Always use TE** unless:
- Using GPUs older than Volta (TE requires CUDA compute capability ≥ 7.0)
- Debugging numerical issues (use `--transformer-impl local` for pure PyTorch)
- Running inference on non-NVIDIA hardware

### TE Version Requirements

| Feature | Minimum TE Version | Recommended |
|---------|-------------------|-------------|
| **Basic TE layers** | 1.0 | 2.9+ |
| **FP8 training** | 1.0 | 2.9+ |
| **Symmetric all-reduce** | 2.3.0 | 2.9+ |
| **NCCL userbuffers** | 2.9.0 | 2.9+ |
| **FSDP2 FP8 param gather** | 2.1.0+ | 2.9+ |
| **TEFusedMLP** | 1.13.0 | 2.9+ |
| **Blockwise FP8** | 2.3.0 | 2.9+ |
| **MXFP8 (Blackwell)** | 2.1.0 | 2.9+ |

**Check your version**:
```bash
python -c "import transformer_engine; print(transformer_engine.__version__)"
```

**Install/upgrade**:
```bash
pip install transformer-engine[pytorch]>=2.9.0
```

---

## Quick Start

### Minimal TE Configuration

**Recommended for all users**:
```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --attention-backend auto \
    --use-mcore-models \
    --bf16 \
    ...
```

**That's it!** TE will auto-select the best optimizations.

### FP8 Training (H100/H200 only)

```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --fp8-format hybrid \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
    --attention-backend auto \
    --use-mcore-models \
    --bf16 \
    ...
```

### Maximum Performance (H100 + Multi-node)

```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --fp8-format hybrid \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
    --fp8-param-gather \
    --tp-comm-overlap \
    --tp-comm-overlap-rs-dgrad \
    --nccl-ub \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --attention-backend auto \
    --use-mcore-models \
    --bf16 \
    ...
```

---

## CLI Flags Reference

### Core TE Flags

#### `--transformer-impl`

**Values**: `local`, `transformer_engine`, `inference_optimized`
**Default**: `transformer_engine`
**Location**: `megatron/training/arguments.py:1345`

Selects transformer implementation:
- **`transformer_engine`** (recommended): Use TE layers with fused ops and FP8 support
- **`local`**: Use Megatron native PyTorch layers (for debugging)
- **`inference_optimized`**: Optimized for inference (experimental)

```bash
--transformer-impl transformer_engine
```

---

### FP8 Training Flags

#### `--fp8-format`

**Values**: `e4m3`, `hybrid`
**Default**: `None` (FP8 disabled)
**Location**: `megatron/training/arguments.py:1315`

FP8 format for weights and activations:
- **`e4m3`**: Use E4M3 for both forward and backward (fastest, slightly less accurate)
- **`hybrid`**: E4M3 for forward, E5M2 for backward gradients (recommended, best accuracy)

```bash
--fp8-format hybrid
```

**Hardware requirement**: Hopper (H100/H200) or Ada (RTX 4090, L40S)

**Expected speedup**: 1.5-2× throughput vs BF16

---

#### `--fp8-recipe`

**Values**: `delayed`, `tensorwise`, `mxfp8`, `blockwise`, `custom`
**Default**: `delayed`
**Location**: `megatron/training/arguments.py:1320`

FP8 quantization recipe:

| Recipe | TE Version | Use Case | Accuracy | Speed |
|--------|------------|----------|----------|-------|
| **`delayed`** | 1.0+ | Production (default) | Best | Good |
| **`tensorwise`** | 2.2.0+ | Experimental | Good | Better |
| **`mxfp8`** | 2.1.0+ | Blackwell GPUs | TBD | Best (on Blackwell) |
| **`blockwise`** | 2.3.0+ | Research | Good | Good |
| **`custom`** | 1.0+ | Custom quantizer | Varies | Varies |

```bash
--fp8-recipe delayed  # Recommended for production
```

**See**: [10-fp8-training.md](10-fp8-training.md) for recipe comparison.

---

#### `--fp8-amax-history-len`

**Type**: int
**Default**: `1` (TE < 2.0), `1024` (recommended for TE ≥ 2.0)
**Location**: `megatron/training/arguments.py:1335`

Number of steps to track for AMAX (absolute maximum) scaling:
- **1**: Use only current step's AMAX (fast updates, can be unstable)
- **1024**: Track 1024 steps (stable, recommended)
- **Higher**: More stable but slower to adapt

```bash
--fp8-amax-history-len 1024
```

---

#### `--fp8-amax-compute-algo`

**Values**: `most_recent`, `max`
**Default**: `most_recent`
**Location**: `megatron/training/arguments.py:1338`

How to compute AMAX from history:
- **`most_recent`**: Use most recent AMAX (faster adaptation)
- **`max`**: Use maximum AMAX over history (more conservative, stable)

```bash
--fp8-amax-compute-algo max  # Recommended for production
```

---

#### `--fp8-param-gather`

**Type**: flag
**Default**: `False`
**Location**: `megatron/training/arguments.py:1348`

Enable FP8 all-gather for distributed optimizer parameters:
- **Enabled**: All-gather params in FP8 (2× faster, requires more memory)
- **Disabled**: All-gather params in BF16

```bash
--fp8-param-gather
```

**Requirements**:
- Must use `--use-distributed-optimizer`
- TE ≥ 2.0 for non-FSDP2
- TE ≥ 2.1 for FSDP2
- `--fp8-recipe delayed` if using CPU offloading

**Performance**: 20-30% faster distributed optimizer

---

#### `--fp8-margin`

**Type**: int
**Default**: `0`
**Location**: `megatron/training/arguments.py:1329`

Margin (in units) added to AMAX for safety:
- **0**: No margin (recommended)
- **Positive**: Add margin to prevent clipping (conservative)

```bash
--fp8-margin 0
```

---

#### `--fp8-interval`

**Type**: int
**Default**: `1`
**Location**: `megatron/training/arguments.py:1332`

**DEPRECATED**: Ignored in TE ≥ 2.0. Scaling updates happen every step.

---

#### `--fp8-quantizer-factory`

**Type**: str (Python path to factory function)
**Default**: `None`
**Location**: `megatron/training/arguments.py:1324`

Custom quantizer factory for `--fp8-recipe custom`:
```bash
--fp8-recipe custom \
--fp8-quantizer-factory my_module.my_quantizer_factory
```

**Advanced use only**. See TE docs for custom quantizer API.

---

### Communication Optimization Flags

#### `--tp-comm-overlap`

**Type**: flag
**Default**: `False`
**Location**: `megatron/training/arguments.py:2092`

Enable communication overlap for tensor parallelism:
- Overlaps TP all-gather/reduce-scatter with computation
- Uses TE userbuffers for zero-copy communication

```bash
--tp-comm-overlap
```

**Requirements**:
- `--transformer-impl transformer_engine`
- TE ≥ 2.0

**Performance**: 10-20% speedup for TP ≥ 2

---

#### `--tp-comm-overlap-cfg`

**Type**: str (JSON config)
**Default**: `None`
**Location**: `megatron/training/arguments.py:2094`

Fine-grained control over TP communication overlap:
```bash
--tp-comm-overlap-cfg '{"method": "ring", "num_sm": 2}'
```

**Options**:
- `method`: `bulk`, `ring`, `pipeline`
- `num_sm`: Number of SMs for communication (default: auto)

**Advanced use only**.

---

#### `--tp-comm-overlap-rs-dgrad`

**Type**: flag
**Default**: `False`
**Location**: `megatron/training/arguments.py:2104`

Overlap reduce-scatter for data-gradient (dgrad) in TP:
```bash
--tp-comm-overlap-rs-dgrad
```

**Performance**: Additional 5-10% speedup with `--tp-comm-overlap`

---

#### `--nccl-ub`

**Type**: flag
**Default**: `False`

Enable NCCL userbuffers for TE communication:
- Reduces SM usage for communication (more SMs for compute)
- Requires NCCL with userbuffer support

```bash
--nccl-ub
```

**Requirements**:
- TE ≥ 2.9
- NCCL compiled with userbuffer support (NGC containers have this)

**Performance**: 5-10% speedup for multi-node training

---

#### `--delay-wgrad-compute`

**Type**: flag
**Default**: `False`
**Location**: `megatron/training/arguments.py:3180`

Delay weight gradient (wgrad) computation to overlap with communication:
```bash
--delay-wgrad-compute
```

**Requirements**:
- `--transformer-impl transformer_engine`
- TE ≥ 2.7 (for gradient accumulation fusion)
- TE ≥ 2.8 (for `--overlap-grad-reduce`)

**Performance**: 10-15% speedup for large TP

---

### CUDA Graph Flags

#### `--cuda-graph-impl`

**Values**: `none`, `local`, `transformer_engine`
**Default**: `none`
**Location**: `megatron/training/arguments.py:1416`

CUDA graph capture implementation:
- **`none`**: Disabled (default)
- **`local`**: Megatron native CUDA graphs
- **`transformer_engine`**: TE `make_graphed_callables()` (recommended)

```bash
--cuda-graph-impl transformer_engine
```

**Performance**: 5-10% speedup for inference, small models

**Note**: CUDA graphs have strict memory layout requirements. May not work with all configurations.

---

#### `--external-cuda-graph`

**Type**: flag
**Default**: `False`
**Location**: `megatron/training/arguments.py:1413`

**DEPRECATED**: Use `--cuda-graph-impl transformer_engine` instead.

---

### Miscellaneous TE Flags

#### `--te-rng-tracker`

**Type**: flag
**Default**: `False`
**Location**: `megatron/training/arguments.py:1375`

Use TE's RNG tracker instead of Megatron's:
```bash
--te-rng-tracker
```

**When to use**: If you see dropout-related numerical differences when switching to TE.

**Default behavior**: Megatron's RNG tracker is used even with TE.

---

## Configuration Recipes by Model Type

### GPT-Style Models (GPT-3, LLaMA, Qwen, DeepSeek)

#### LLaMA 3 8B (Single Node, 8× H100)

**Configuration**: `examples/llama/train_llama3_8b_h100_fp8.sh`

```bash
# Model architecture
--use-mcore-models \
--num-layers 32 \
--hidden-size 4096 \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--group-query-attention \
--num-query-groups 8 \
--seq-length 8192 \
--max-position-embeddings 8192 \
--position-embedding-type rope \
--rotary-base 1000000 \
--swiglu \
--disable-bias-linear \
--untie-embeddings-and-output-weights \
\
# Transformer Engine + FP8
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max \
--fp8-param-gather \
--attention-backend fused \
\
# Training
--micro-batch-size 1 \
--global-batch-size 128 \
--bf16 \
--cross-entropy-loss-fusion \
\
# Parallelism (single node)
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--sequence-parallel \
\
# Distributed optimizer
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

**Expected MFU**: 45-47% on H100

---

#### GPT-3 175B (Multi-Node, 128× A100)

**Configuration**: `examples/gpt3/train_gpt3_175b_distributed.sh` (adapted for TE)

```bash
# Model architecture
--num-layers 96 \
--hidden-size 12288 \
--num-attention-heads 96 \
--seq-length 2048 \
--max-position-embeddings 2048 \
\
# Transformer Engine (no FP8 on A100)
--transformer-impl transformer_engine \
--attention-backend auto \
\
# Training
--micro-batch-size 1 \
--global-batch-size 1536 \
--fp16 \
\
# Parallelism (16 nodes × 8 GPUs)
--tensor-model-parallel-size 8 \
--pipeline-model-parallel-size 16 \
--sequence-parallel \
\
# Communication overlap
--tp-comm-overlap \
--overlap-grad-reduce \
--use-distributed-optimizer
```

**Expected speedup**: 1.2-1.3× vs native Megatron on A100 (without FP8)

---

### T5-Style Models (Encoder-Decoder)

#### T5 220M (8× V100)

**Configuration**: `examples/t5/train_t5_220m_distributed.sh` (adapted for TE)

```bash
# Model architecture
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
--ffn-hidden-size 2048 \
--encoder-num-layers 12 \
--decoder-num-layers 12 \
--seq-length 512 \
--max-position-embeddings 512 \
\
# Transformer Engine (V100 compatible)
--transformer-impl transformer_engine \
--attention-backend auto \
\
# Training (V100 = FP16 only)
--micro-batch-size 16 \
--global-batch-size 128 \
--fp16 \
\
# Parallelism
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1
```

**Note**: V100 doesn't support BF16 or FP8. Use `--fp16`.

---

### MoE Models (Mixtral, DeepSeek-V3)

#### Mixtral 8x7B (64× A100)

**Configuration**: `examples/mixtral/train_mixtral_8x7b_distributed.sh` (adapted for TE)

```bash
# Model architecture
--use-mcore-models \
--num-layers 32 \
--hidden-size 4096 \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--group-query-attention \
--num-query-groups 8 \
--seq-length 4096 \
--max-position-embeddings 32768 \
--swiglu \
--disable-bias-linear \
--normalization RMSNorm \
--position-embedding-type rope \
\
# MoE configuration
--num-experts 8 \
--moe-router-topk 2 \
--moe-grouped-gemm \
--moe-token-dispatcher-type alltoall \
\
# Transformer Engine
--transformer-impl transformer_engine \
--attention-backend auto \
\
# Training
--micro-batch-size 1 \
--global-batch-size 256 \
--bf16 \
\
# Parallelism (8 nodes × 8 GPUs)
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 4 \
--expert-model-parallel-size 8 \
--sequence-parallel \
\
# Distributed optimizer + overlap
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

**MoE-specific**: TE provides `TEGroupedLinear` for experts (use `--moe-grouped-gemm`).

---

### Long-Context Models (8K+ Sequences)

#### LLaMA with 32K Context (64× H100)

```bash
# Model architecture
--use-mcore-models \
--num-layers 32 \
--hidden-size 4096 \
--num-attention-heads 32 \
--seq-length 32768 \
--max-position-embeddings 32768 \
--position-embedding-type rope \
--rotary-base 1000000 \
--swiglu \
\
# Transformer Engine + FP8
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max \
--attention-backend flash \  # Flash Attention required for long context
\
# Training
--micro-batch-size 1 \
--global-batch-size 512 \
--bf16 \
\
# Parallelism (8 nodes × 8 GPUs)
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 4 \
--context-parallel-size 4 \  # Critical for long sequences
--sequence-parallel \
\
# Communication overlap
--tp-comm-overlap \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

**Key**: Use `--context-parallel-size` to distribute long sequences.

---

## Hardware-Specific Guides

### Hopper GPUs (H100, H200)

**Optimal Configuration**:

```bash
# Always use FP8 on Hopper
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max \
--fp8-param-gather \
\
# Enable all TE optimizations
--tp-comm-overlap \
--tp-comm-overlap-rs-dgrad \
--nccl-ub \  # Requires TE ≥ 2.9
--delay-wgrad-compute \
\
# Use Flash Attention 3 (Hopper-optimized)
--attention-backend auto \  # Auto-selects FA3 if available
\
# BF16 base precision (FP8 on top)
--bf16
```

**Expected Performance**:
- **MFU**: 45-47% (state-of-the-art)
- **Speedup vs A100 BF16**: 2-2.5×
- **Speedup vs H100 BF16**: 1.5-2×

**Environment Variables** (optional tuning):
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Serialize kernel launches
export NVTE_FWD_LAYERNORM_SM_MARGIN=16  # TE LayerNorm tuning
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

---

### Ampere GPUs (A100, A30)

**Optimal Configuration**:

```bash
# Use TE but without FP8 (A100 has limited FP8 support)
--transformer-impl transformer_engine \
--attention-backend auto \
\
# BF16 training (A100 has good BF16 performance)
--bf16 \
\
# Enable TE optimizations (no FP8-specific ones)
--tp-comm-overlap \
--delay-wgrad-compute \
\
# Distributed optimizer
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

**Expected Performance**:
- **MFU**: 35-40%
- **Speedup vs native Megatron**: 1.2-1.4×

**Why not FP8 on A100?**
- A100 has limited FP8 tensor core support (only for inference in some models)
- TE FP8 training targets Hopper architecture

---

### Volta GPUs (V100)

**Optimal Configuration**:

```bash
# Use TE (V100 compatible)
--transformer-impl transformer_engine \
--attention-backend fused \  # V100 doesn't support Flash Attention
\
# FP16 training (V100 doesn't support BF16)
--fp16 \
\
# Simpler config (V100 has fewer TE optimizations)
--use-distributed-optimizer \
--overlap-grad-reduce
```

**Limitations on V100**:
- No BF16 support (must use `--fp16`)
- No Flash Attention (use `--attention-backend fused`)
- No FP8
- Limited TE optimizations (no userbuffers, no TP comm overlap)

**Expected Performance**:
- **MFU**: 25-30%
- **Speedup vs native Megatron**: 1.1-1.2× (minimal)

**Recommendation**: If possible, upgrade to A100 or H100 for significant TE benefits.

---

### Multi-Node Networking

#### InfiniBand (Recommended)

```bash
# NCCL tuning for IB
export NCCL_IB_TIMEOUT=19
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # Adjust to your IB adapters

# TE-specific
--nccl-ub \  # NCCL userbuffers (TE ≥ 2.9)
--tp-comm-overlap
```

#### Ethernet (GbE/100GbE)

```bash
# Disable IB-specific optimizations
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # Your network interface

# TE comm overlap still helps
--tp-comm-overlap
```

---

## FP8 Configuration Deep Dive

### FP8 Format Selection

#### `hybrid` (Recommended)

**Use for**: Production training, best accuracy

```bash
--fp8-format hybrid \
--fp8-recipe delayed \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max
```

**Behavior**:
- Forward pass: E4M3 (8-bit mantissa, 3-bit exponent)
- Backward pass gradients: E5M2 (5-bit mantissa, 2-bit exponent)
- Master weights: BF16

**Accuracy**: Matches BF16 within 1-2% on most models

---

#### `e4m3` (Fastest)

**Use for**: Maximum speed, can tolerate slight accuracy drop

```bash
--fp8-format e4m3 \
--fp8-recipe delayed \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max
```

**Behavior**:
- Forward and backward: E4M3
- Faster than hybrid (no format conversion)

**Accuracy**: 2-5% drop vs BF16 (model-dependent)

---

### FP8 Recipe Comparison

| Recipe | Scaling Update | Communication | Accuracy | Speed | Use Case |
|--------|----------------|---------------|----------|-------|----------|
| **`delayed`** | Every step | Standard | Best | Good | Production (default) |
| **`tensorwise`** | Per-tensor | Standard | Good | Better | Experimental |
| **`mxfp8`** | Microscaling | Reduced | TBD | Best | Blackwell only |
| **`blockwise`** | Per-block | Standard | Good | Good | Research |

**Recommendation**: Start with `delayed`. Try `tensorwise` if you want faster convergence.

---

### AMAX History Tuning

**Default (Old TE)**: `--fp8-amax-history-len 1`
**Recommended (TE ≥ 2.0)**: `--fp8-amax-history-len 1024`

**Effect of history length**:
- **1**: Fast adaptation, can be unstable (spikes cause clipping)
- **1024**: Stable, smooth scaling updates
- **Higher (2048+)**: Very stable, slower to adapt to distribution shifts

**When to increase**:
- Seeing NaN/Inf during training
- Training loss is noisy

**When to decrease**:
- Want faster adaptation to learning rate changes
- Training is stable

---

### FP8 Param Gather

**Flag**: `--fp8-param-gather`

**Effect**: All-gather distributed optimizer parameters in FP8 instead of BF16.

**Requirements**:
```bash
--use-distributed-optimizer \
--fp8-param-gather \
--fp8-recipe delayed  # If using CPU offload
```

**Performance**:
- **Communication**: 2× faster (FP8 is half the size of BF16)
- **Memory**: Slightly higher (FP8 buffers)
- **Accuracy**: No impact (params are decompressed before use)

**Speedup**: 20-30% faster distributed optimizer step

---

## Communication Optimization

### Tensor Parallelism Overlap

**Basic Overlap**:
```bash
--tp-comm-overlap
```

**Advanced Overlap (reduce-scatter for dgrad)**:
```bash
--tp-comm-overlap \
--tp-comm-overlap-rs-dgrad
```

**With NCCL Userbuffers** (TE ≥ 2.9):
```bash
--tp-comm-overlap \
--nccl-ub
```

**Performance Impact**:
- Basic overlap: 10-20% speedup for TP ≥ 2
- + RS dgrad: Additional 5-10%
- + NCCL UB: Additional 5-10%

---

### Data Parallelism Overlap

**Gradient Reduce Overlap**:
```bash
--overlap-grad-reduce
```

**Parameter Gather Overlap** (distributed optimizer):
```bash
--use-distributed-optimizer \
--overlap-param-gather
```

**Both**:
```bash
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

**Performance**: Hides 80-90% of DP communication

---

### Delayed Weight Gradient

**Flag**: `--delay-wgrad-compute`

**Effect**: Delay wgrad computation to overlap with grad reduction.

**Requirements**:
- TE ≥ 2.7 (for gradient accumulation fusion)
- TE ≥ 2.8 (for `--overlap-grad-reduce` compatibility)

```bash
--delay-wgrad-compute \
--overlap-grad-reduce
```

**Performance**: 10-15% speedup for large TP

---

## Troubleshooting Common Issues

### Issue 1: "ImportError: No module named 'transformer_engine'"

**Symptom**:
```
ModuleNotFoundError: No module named 'transformer_engine'
```

**Solution**:
```bash
pip install transformer-engine[pytorch]
```

**Verify**:
```bash
python -c "import transformer_engine; print(transformer_engine.__version__)"
```

---

### Issue 2: "FP8 not supported on this GPU"

**Symptom**:
```
RuntimeError: FP8 training is only supported on Hopper (H100/H200) and Ada (RTX 4090) GPUs
```

**Solution**: Remove FP8 flags for A100/V100:
```bash
# Remove these:
# --fp8-format hybrid
# --fp8-amax-history-len 1024
# --fp8-param-gather

# Keep TE without FP8:
--transformer-impl transformer_engine \
--bf16
```

---

### Issue 3: "FSDP2 FP8 param gather not supported in TE 2.0"

**Symptom**:
```
Warning: FSDP2 FP8 param gather is not supported yet in TE 2.0, will fallback to bf16
```

**Solution**: Upgrade TE:
```bash
pip install --upgrade transformer-engine[pytorch]>=2.1.0
```

Or disable FP8 param gather:
```bash
# Remove --fp8-param-gather
```

---

### Issue 4: "tp-comm-overlap requires transformer_engine"

**Symptom**:
```
AssertionError: --tp-comm-overlap requires --transformer-impl transformer_engine
```

**Solution**: Add TE:
```bash
--transformer-impl transformer_engine \
--tp-comm-overlap
```

---

### Issue 5: "delay-wgrad-compute requires TE ≥ 2.7"

**Symptom**:
```
AssertionError: delay_wgrad_compute requires TE >= 2.7.0 for gradient accumulation fusion
```

**Solution**: Upgrade TE or disable:
```bash
# Option 1: Upgrade
pip install --upgrade transformer-engine[pytorch]>=2.7.0

# Option 2: Disable
# Remove --delay-wgrad-compute
```

---

### Issue 6: "NaN/Inf in FP8 training"

**Symptom**: Loss becomes NaN after a few steps with FP8.

**Diagnosis**:
```bash
# Add gradient clipping
--clip-grad 1.0

# Increase AMAX history
--fp8-amax-history-len 2048

# Use max instead of most_recent
--fp8-amax-compute-algo max

# Switch to hybrid format
--fp8-format hybrid  # instead of e4m3
```

---

### Issue 7: "TE version mismatch warnings"

**Symptom**:
```
Warning: TE version 1.8 is older than recommended 2.9+. Some features may not work.
```

**Solution**: Upgrade to latest TE:
```bash
pip install --upgrade transformer-engine[pytorch]
```

**Check what's available**:
```python
import transformer_engine
print(f"TE version: {transformer_engine.__version__}")

# Check for specific features
from transformer_engine.pytorch import make_graphed_callables  # TE ≥ 1.0
# from transformer_engine.pytorch import symmetric_memory_efficient_all_reduce  # TE ≥ 2.3
```

---

## Performance Tuning Checklist

### Pre-Training Checklist

Before starting multi-day training, verify:

- [ ] **TE installed**: `python -c "import transformer_engine; print(transformer_engine.__version__)"` shows ≥ 2.9
- [ ] **Correct GPU**: H100/H200 for FP8, A100 for BF16 TE
- [ ] **TE enabled**: `--transformer-impl transformer_engine`
- [ ] **FP8 configured** (H100 only):
  - [ ] `--fp8-format hybrid`
  - [ ] `--fp8-amax-history-len 1024`
  - [ ] `--fp8-amax-compute-algo max`
  - [ ] `--fp8-param-gather` (if using distributed optimizer)
- [ ] **Communication overlap**:
  - [ ] `--tp-comm-overlap` (if TP ≥ 2)
  - [ ] `--overlap-grad-reduce`
  - [ ] `--overlap-param-gather` (if distributed optimizer)
- [ ] **Attention backend**: `--attention-backend auto` (or `flash` for long sequences)
- [ ] **Environment variables set** (if H100):
  - [ ] `CUDA_DEVICE_MAX_CONNECTIONS=1`
- [ ] **Test on 1 GPU**: Verify no import errors
- [ ] **Test on 8 GPUs**: Verify distributed works
- [ ] **Check MFU**: Should be 40-47% on H100, 35-40% on A100

---

### Performance Validation

**Measure MFU** (Model FLOP Utilization):

Run with profiling:
```bash
--profile \
--profile-step-start 10 \
--profile-step-end 12 \
--log-throughput
```

**Expected MFU**:
- H100 + FP8 + TE: **45-47%**
- H100 + BF16 + TE: **30-35%**
- A100 + BF16 + TE: **35-40%**
- A100 + BF16 + Native: **25-30%**

**If MFU is low (<30% on H100)**:
1. Check FP8 is actually enabled (look for "Using FP8" in logs)
2. Verify TE version ≥ 2.9
3. Check communication overlap is working
4. Profile to find bottleneck

---

### Iterative Tuning

**Start conservative, add optimizations incrementally**:

1. **Baseline** (TE only):
   ```bash
   --transformer-impl transformer_engine --bf16
   ```

2. **+ FP8** (H100 only):
   ```bash
   --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max
   ```

3. **+ TP overlap**:
   ```bash
   --tp-comm-overlap --tp-comm-overlap-rs-dgrad
   ```

4. **+ DP overlap**:
   ```bash
   --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
   ```

5. **+ Advanced**:
   ```bash
   --delay-wgrad-compute --nccl-ub --fp8-param-gather
   ```

**Measure MFU at each step**. If MFU decreases, revert last change.

---

## Summary

### Quick Configuration Templates

**H100 (FP8, Maximum Performance)**:
```bash
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max \
--fp8-param-gather \
--tp-comm-overlap \
--tp-comm-overlap-rs-dgrad \
--nccl-ub \
--delay-wgrad-compute \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \
--attention-backend auto \
--bf16
```

**A100 (BF16, Good Performance)**:
```bash
--transformer-impl transformer_engine \
--tp-comm-overlap \
--delay-wgrad-compute \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \
--attention-backend auto \
--bf16
```

**V100 (FP16, Basic)**:
```bash
--transformer-impl transformer_engine \
--attention-backend fused \
--use-distributed-optimizer \
--overlap-grad-reduce \
--fp16
```

---

### Key Takeaways

1. **Always use TE** (`--transformer-impl transformer_engine`) unless debugging
2. **FP8 on Hopper** gives 2× speedup - always enable it on H100/H200
3. **Communication overlap** is critical for multi-GPU/multi-node (10-30% speedup)
4. **TE ≥ 2.9** unlocks best performance (NCCL userbuffers, FP8 param gather)
5. **Start conservative**, add optimizations incrementally
6. **Measure MFU** to validate performance

---

## Related Documents

- **[08-kernel-selection-guide.md](08-kernel-selection-guide.md)**: Attention backend selection
- **[09-transformer-engine-integration.md](09-transformer-engine-integration.md)**: TE architecture deep dive
- **[10-fp8-training.md](10-fp8-training.md)**: FP8 recipes and quantization details
- **[11-te-optimizations.md](11-te-optimizations.md)**: Advanced TE optimizations
- **[02-communication-overlap.md](02-communication-overlap.md)**: Communication overlap theory

---

**Document Version**: 1.0
**Last Updated**: 2025-12-07
**Estimated Reading Time**: 25 minutes
**Target Audience**: All Megatron + TE users
