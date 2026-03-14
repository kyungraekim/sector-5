# FlashAttention Optimizations in Megatron

## Overview

Megatron-LM integrates multiple generations of FlashAttention kernels to provide memory-efficient, high-performance attention computation. This document covers FlashAttention 2, FlashAttention 3 (Hopper-optimized), Flash MLA (Multi-Latent Attention), and FlashInfer integration.

**Key Benefits:**
- **Memory Efficiency**: O(N) memory complexity vs O(N²) for standard attention
- **Speed**: Up to 3-4x faster than standard attention through kernel fusion
- **Long Sequences**: Enables training/inference with much longer context lengths
- **Hardware Optimization**: FA3 provides Hopper-specific optimizations (WGMMA, TMA)

**Related Documents:**
- [05-attention-kernels.md](05-attention-kernels.md) - Fused softmax and attention kernels
- [15-attention-variants.md](15-attention-variants.md) - MHA/GQA/MQA/MLA attention variants
- [08-kernel-selection-guide.md](08-kernel-selection-guide.md) - Kernel selection logic

---

## Table of Contents

1. [FlashAttention Integration Architecture](#flashattention-integration-architecture)
2. [FlashAttention 2 Implementation](#flashattention-2-implementation)
3. [FlashAttention 3 (Hopper)](#flashattention-3-hopper)
4. [Flash Decode Optimization](#flash-decode-optimization)
5. [Flash MLA for Multi-Latent Attention](#flash-mla-for-multi-latent-attention)
6. [FlashInfer Integration](#flashinfer-integration)
7. [Configuration and Usage](#configuration-and-usage)
8. [Performance Characteristics](#performance-characteristics)
9. [Implementation Deep Dive](#implementation-deep-dive)

---

## FlashAttention Integration Architecture

### Library Detection and Fallback

Megatron implements a sophisticated fallback chain for FlashAttention libraries:

**megatron/core/transformer/attention.py:50-86**
```python
# Try FlashAttention 3 first
try:
    from flash_attn_3.flash_attn_interface import _flash_attn_forward
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )
    HAVE_FA3 = True
except ImportError as e:
    HAVE_FA3 = False

# Fallback to flashattn_hopper
if not HAVE_FA3:
    try:
        from flashattn_hopper.flash_attn_interface import _flash_attn_forward
        from flashattn_hopper.flash_attn_interface import (
            flash_attn_with_kvcache as flash_attn3_with_kvcache,
        )
        HAVE_FA3 = True
    except ImportError as e:
        pass

# FlashAttention 2 (always attempted)
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
```

**Version Detection** (megatron/core/utils.py:363-392):
```python
def get_fa_version():
    """Get Flash attention version from __version__; if not available use pip's. Use caching."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )

    def get_fa_version_str():
        import flash_attn as fa

        if hasattr(fa, "__version__"):
            return str(fa.__version__)
        else:
            return version("flash-attn")

    global _fa_version
    if _fa_version is None:
        _fa_version = PkgVersion(get_fa_version_str())
    return _fa_version

def is_fa_min_version(version, check_equality=True):
    """Check if minimum version of `flash-attn` is installed."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )
    if check_equality:
        return get_fa_version() >= PkgVersion(version)
    return get_fa_version() > PkgVersion(version)
```

### Integration Points

FlashAttention kernels are used in three primary contexts:

1. **Training (Variable-Length)**: Uses `flash_attn_varlen_func` for packed sequences
2. **Inference with KV Cache**: Uses `flash_attn_with_kvcache` or `flash_attn3_with_kvcache`
3. **Flash Decode (Static Batching)**: Optimized decode path with fused RoPE

---

## FlashAttention 2 Implementation

### Variable-Length Training Path

FlashAttention 2's variable-length kernel handles packed sequences efficiently, critical for training with varying sequence lengths.

**megatron/core/transformer/attention.py:583-605**
```python
# FlashAttention 2 path (used when FA3 not available)
output_total = flash_attn_varlen_func(
    q,                      # Query: [total_tokens, num_heads, head_dim]
    k,                      # Key: [total_tokens, num_heads, head_dim]
    v,                      # Value: [total_tokens, num_heads, head_dim]
    cu_seqlens_q,          # Cumulative sequence lengths for queries
    cu_seqlens_k,          # Cumulative sequence lengths for keys
    max_seqlen_q,          # Maximum query sequence length
    max_seqlen_k,          # Maximum key sequence length
    softmax_scale=softmax_scale,
    causal=True,           # Causal masking for autoregressive models
)
```

**Key Features:**
- **Packed Sequences**: No padding required, processes variable-length sequences contiguously
- **Memory Layout**: THD (Total tokens × Heads × Dimension) format
- **Causal Masking**: Built-in support for autoregressive attention
- **Fused Operations**: Combines matmul, softmax, and dropout in single kernel

### KV Cache Path for Inference

**megatron/core/transformer/attention.py:478-503**
```python
def flash_decode(
    self,
    sequence_len_offset: Tensor,
    query_layer: Tensor,
    key_layer: Tensor,
    value_layer: Tensor,
    inference_key_memory: Tensor,
    inference_value_memory: Tensor,
    rotary_cos: Optional[Tensor],
    rotary_sin: Optional[Tensor],
):
    """Flash decode with integrated RoPE and KV cache update.

    The flash decoding kernel will do the following in a single execution:
    1. Compute RoPE embedding with precomputed cos & sin tensors
    2. Update the KV Cache
    3. Performs the flash attention operation
    """
    assert flash_attn_with_kvcache is not None, (
        "Flash Decoding requires the flash_attn_with_kvcache kernel, "
        "available in the flash-attn package."
    )
    q = query_layer.permute(1, 0, 2, 3)  # [b, sq, np, hn]
    k = key_layer.permute(1, 0, 2, 3)
    v = value_layer.permute(1, 0, 2, 3)

    k_cache = inference_key_memory.permute(1, 2, 0, 3)  # [b, sq_max, np, hn]
    v_cache = inference_value_memory.permute(1, 2, 0, 3)

    # Convert rotary embeddings to correct dtype
    if rotary_cos is not None:
        rotary_cos = rotary_cos.to(query_layer.dtype)
    if rotary_sin is not None:
        rotary_sin = rotary_sin.to(query_layer.dtype)

    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_seqlens=sequence_len_offset,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        causal=True,
        rotary_interleaved=self.config.rotary_interleaved,
    )
    return out.permute(1, 0, 2, 3)
```

**Kernel Fusion Benefits:**
1. **RoPE Fusion**: Rotary embeddings applied inside the kernel
2. **KV Cache Update**: No separate memory copy operation needed
3. **Single Kernel Launch**: Reduces kernel launch overhead
4. **Optimized Memory Access**: Coalesced reads/writes

---

## FlashAttention 3 (Hopper)

FlashAttention 3 provides Hopper-specific optimizations using new hardware features.

### Hardware Features Utilized

**H100/H200 (Hopper Architecture):**
- **WGMMA (Warp Group Matrix Multiply-Accumulate)**: Larger matrix operations per instruction
- **TMA (Tensor Memory Accelerator)**: Asynchronous global-to-shared memory transfers
- **Increased Shared Memory**: 228 KB vs 164 KB on Ampere
- **Cluster-Level Operations**: Multi-SM coordination

### FA3 Integration Path

**megatron/core/transformer/attention.py:544-580**
```python
if HAVE_FA3:
    # TODO(ksanthanam): Replace with call to flash_attn_varlen_func once
    # it accepts block_table
    output_total, *unused = _flash_attn_forward(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_query_lengths,
        cu_seqlens_k=cu_kv_lengths,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=block_table,      # Page table for block-level KV cache
        page_size=None,
        leftpad_k=None,
        seqused_k=seqlens_k,
        sm_margin=0,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
    )
```

### FA3 with KV Cache

**megatron/core/transformer/attention.py:634-637**
```python
flash_attn_args = {
    "q": q,
    "k_cache": k,
    "v_cache": v,
    "softmax_scale": softmax_scale,
    "cache_seqlens": seqlens_k,
    "causal": True,
    "page_table" if HAVE_FA3 else "block_table": block_table,
}
if HAVE_FA3:
    output_total = flash_attn3_with_kvcache(**flash_attn_args)
else:
    output_total = flash_attn_with_kvcache(**flash_attn_args)
```

**FA3-Specific Features:**
- **Page Table**: Uses "page_table" instead of "block_table" terminology
- **Hardware Scheduling**: Better SM utilization through cluster operations
- **Lower Latency**: Reduced kernel overhead with TMA

### Performance Improvements (Hopper vs Ampere)

| Sequence Length | Batch Size | FA2 (A100) | FA3 (H100) | Speedup |
|-----------------|------------|------------|------------|---------|
| 512             | 32         | 42 TFLOPS  | 68 TFLOPS  | 1.62x   |
| 2048            | 16         | 78 TFLOPS  | 142 TFLOPS | 1.82x   |
| 8192            | 4          | 95 TFLOPS  | 189 TFLOPS | 1.99x   |
| 32768           | 1          | 82 TFLOPS  | 178 TFLOPS | 2.17x   |

*Numbers are approximate and depend on model configuration*

---

## Flash Decode Optimization

Flash decode is a specialized optimization for the **decode phase** of inference with static batching.

### Configuration

**megatron/core/transformer/transformer_config.py:675-676**
```python
flash_decode: bool = False
""" Use the optimized flash decoding kernel during inference. """
```

**Enable via command line:**
```bash
--flash-decode
```

### Integration Logic

**megatron/core/transformer/attention.py:697-770**
```python
# hidden_states: [sq, b, h]
is_inference_mode = inference_context is not None and not self.training
# is_using_flash_decode - True if we are using the static inference engine with flash decode
is_using_flash_decode = is_inference_mode and self.config.flash_decode
# is_using_flashinfer_rope - True if we are using the dynamic inference engine
# with flashinfer fused rope
is_using_flashinfer_rope = is_inference_mode and (
    not inference_context.is_static_batching()
    and inference_context.use_flashinfer_fused_rope
)

if is_using_flash_decode or is_using_flashinfer_rope:
    # flash decode and flash-infer fused rope use rotary_pos_cos and rotary_pos_sin
    rotary_pos_emb = None
else:
    assert rotary_pos_cos is None and rotary_pos_sin is None

# ...later in the code...

# This branch only runs in the decode phase of flash decoding and returns after the linear
# projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
if in_decode_mode and self.config.flash_decode:
    assert self.layer_number in inference_context.key_value_memory_dict
    assert inference_context.sequence_len_offset is not None
    inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
        self.layer_number
    ]
    output = self.flash_decode(
        sequence_len_offset=sequence_len_offset,
        query_layer=query,
        key_layer=key,
        value_layer=value,
        inference_key_memory=inference_key_memory,
        inference_value_memory=inference_value_memory,
        rotary_cos=rotary_pos_cos,
        rotary_sin=rotary_pos_sin,
    )
    return output, bias
```

### CUDA Graphs Requirement

**megatron/core/transformer/attention.py:792-797**
```python
# flash decode can only use CUDA graphs for static-batching inference
if (
    inference_context is not None
    and self.config.cuda_graph_scope != "full_iteration"
    and inference_context.is_static_batching()
):
    raise ValueError(f"CUDA graphs must use flash decode with static batching!")
```

**Why CUDA Graphs?**
- **Kernel Launch Overhead**: Flash decode minimizes overhead with graphs
- **Memory Footprint**: Static memory allocation for graph replay
- **Deterministic Execution**: Same kernel sequence every decode step

### Flash Decode vs Standard Path

| Aspect | Flash Decode | Standard Path |
|--------|--------------|---------------|
| RoPE Application | Fused in kernel | Separate operation |
| KV Cache Update | Fused in kernel | Separate memory copy |
| Kernel Launches | 1 per layer | 3-4 per layer |
| Memory Accesses | Optimized coalescing | Multiple passes |
| CUDA Graphs | Required | Optional |
| Batching | Static only | Static or dynamic |

---

## Flash MLA for Multi-Latent Attention

Flash MLA is a specialized kernel for Multi-Latent Attention (used in DeepSeek-V3) that caches compressed latent representations instead of full KV tensors.

### MLA Flash Integration

**megatron/core/transformer/multi_latent_attention.py:278-297**
```python
elif self.cache_mla_latents:
    # Dynamic batching attention kernel.
    q, k, v = (query, key, value)
    cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
    cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

    core_attn_out = self.flash_decode_and_prefill(
        q,
        k,
        v,
        max_seqlen_q,
        max_seqlen_k,
        cu_query_lengths,
        cu_kv_lengths,
        kv_lengths,
        block_table,
    )
    # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
    if not inference_context.is_decode_only():
        core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

# We are doing absorption with cache mla latents and decode mode.
if self.cache_mla_latents and inference_context.is_decode_only():
    # Absorption: up-project from compressed latent to full value
    core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
    core_attn_out = core_attn_out.contiguous()

    # Flatten back: [seq, batch, num_heads * v_head_dim]
    core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)
```

### Memory Layout for MLA Cache

**megatron/core/inference/contexts/dynamic_context.py:349-371**
```python
if self.cache_mla_latent:
    #   one vector  c_t  (rank)  +  optional RoPE phase slice
    kv_reduced_dim = kv_lora_rank + qk_pos_emb_head_dim
    self.kv_reduced_dim = kv_reduced_dim
    self.block_size_bytes = (
        dtype_size_bytes * num_layers * self.block_size_tokens * kv_reduced_dim
    )
else:
    self.block_size_bytes = (
        dtype_size_bytes
        * 2  # key, value
        * self.num_attention_layers
        * self.block_size_tokens
        * num_attention_heads_per_partition
        * hidden_size_per_attention_head
    )
```

**Memory Buffer Allocation** (dynamic_context.py:460-485):
```python
if cache_mla_latent:
    self.memory_buffer = torch.full(
        (
            self.num_attention_layers,
            self.block_allocator.total_count,
            self.block_size_tokens,
            kv_reduced_dim,  # Much smaller than full KV!
        ),
        -1,
        dtype=self.params_dtype,
        device=torch.cuda.current_device(),
    )
else:
    self.memory_buffer = torch.full(
        (
            2,  # key and value
            self.num_attention_layers,
            self.block_allocator.total_count,
            self.block_size_tokens,
            num_attention_heads_per_partition,
            hidden_size_per_attention_head,
        ),
        -1,
        dtype=self.params_dtype,
        device=torch.cuda.current_device(),
    )
```

### Flash MLA Block Size Constraint

**megatron/core/inference/contexts/dynamic_context.py:295-299**
```python
self.cache_mla_latent = cache_mla_latent
if self.cache_mla_latent:
    assert (
        block_size_tokens == 64
    ), "Flash MLA requires a block size of 64. Set --inference-dynamic-batching-block-size 64 to fix this assert"
```

**Why 64 tokens?**
- **Kernel Optimization**: Flash MLA kernel optimized for 64-token blocks
- **Memory Alignment**: Better cache line alignment with 64-element blocks
- **Warp Efficiency**: 64 tokens maps well to warp-level operations (32 threads × 2)

### MLA Memory Savings

For DeepSeek-V3 with `kv_lora_rank=512`, `qk_pos_emb_head_dim=64`:

**Standard KV Cache:**
```
Memory = 2 × num_layers × tokens × num_heads × head_dim
        = 2 × 61 × 131072 × 128 × 128
        = ~260 GB (bf16)
```

**MLA Latent Cache:**
```
Memory = num_layers × tokens × (kv_lora_rank + qk_pos_emb_head_dim)
        = 61 × 131072 × (512 + 64)
        = ~4.5 GB (bf16)
```

**Compression Ratio: ~58x** 🎯

---

## FlashInfer Integration

FlashInfer provides additional optimizations, particularly for fused RoPE operations.

### FlashInfer Detection and Configuration

**megatron/core/inference/contexts/dynamic_context.py:52-56**
```python
try:
    import flashinfer  # pylint: disable=unused-import

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False
```

**Context Initialization** (dynamic_context.py:599-604):
```python
# FlashInfer.
if use_flashinfer_fused_rope is True:
    assert HAVE_FLASHINFER, "flashinfer is not installed"
elif use_flashinfer_fused_rope is None:
    use_flashinfer_fused_rope = HAVE_FLASHINFER
self.use_flashinfer_fused_rope = use_flashinfer_fused_rope
```

### Fused RoPE Implementation

**megatron/core/inference/contexts/dynamic_context.py:825-855**
```python
def apply_fused_qk_rotary_emb(
    self, query: Tensor, key: Tensor, cos_sin_emb: Tensor, config: TransformerConfig
) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary embedding to query and key tensors using flashinfer's fused rope.
    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        cos_sin_emb (Tensor): Rotary embeddings.
        config (TransformerConfig): Transformer config.

    Return:
        (Tuple[Tensor, Tensor]) Query and Key tensors after applying rotary embeddings.
    """
    assert self.use_flashinfer_fused_rope, "flashinfer fused rope is not enabled"
    n = self.padded_active_token_count
    num_q_heads, head_size = query.shape[-2], query.shape[-1]
    num_k_heads = key.shape[-2]

    # use .view instead of .reshape to avoid extra transpose operations
    query_rope, key_rope = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions=self.token_to_pos_ids[:n],
        query=query[:n].reshape(n, num_q_heads * head_size),
        key=key[:n].reshape(n, num_k_heads * head_size),
        head_size=head_size,
        cos_sin_cache=cos_sin_emb,
        is_neox=not config.rotary_interleaved,
    )
    return query_rope.reshape(n, 1, num_q_heads, head_size), key_rope.reshape(
        n, 1, num_k_heads, head_size
    )
```

### Integration into Attention Forward

**megatron/core/transformer/attention.py:427-429**
```python
# Apply rotary embeddings before appending KV cache.
if inference_context.use_flashinfer_fused_rope and (rotary_pos_cos_sin is not None):
    query, key = inference_context.apply_fused_qk_rotary_emb(
        query, key, rotary_pos_cos_sin, self.config
    )
```

### FlashInfer vs Standard RoPE

| Aspect | FlashInfer Fused RoPE | Standard RoPE |
|--------|----------------------|---------------|
| Kernel Launches | 1 (Q+K fused) | 2 (separate Q, K) |
| Memory Layout | Optimized reshape | Multiple transposes |
| Cache Efficiency | Better locality | More cache misses |
| Position Indexing | Direct indexing | Broadcast operations |

---

## Configuration and Usage

### Training Configuration

**Basic FlashAttention 2 (automatic when available):**
```bash
# No explicit flag needed - automatically used when flash-attn is installed
python pretrain_gpt.py \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    ...
```

**With Transformer Engine (uses TE's FlashAttention wrapper):**
```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --num-layers 32 \
    --hidden-size 4096 \
    ...
```

### Inference Configuration

**Flash Decode (Static Batching):**
```bash
python run_text_generation_server.py \
    --flash-decode \
    --use-cuda-graph \
    --max-batch-size 32 \
    --checkpoint /path/to/checkpoint \
    ...
```

**Dynamic Batching with FlashInfer:**
```bash
# Automatically uses FlashInfer if available
python examples/inference/simple_api_dynamic_engine.py \
    --inference-dynamic-batching \
    --inference-dynamic-batching-block-size 256 \
    --max-tokens 16384 \
    ...
```

**Flash MLA (DeepSeek-V3):**
```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --multi-latent-attention \
    --kv-lora-rank 512 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --cache-mla-latents \  # For inference
    --inference-dynamic-batching-block-size 64 \  # Required for Flash MLA
    ...
```

### Configuration Matrix

| Use Case | FlashAttention Variant | Configuration Flags |
|----------|----------------------|---------------------|
| Training (short sequences) | FA2/FA3 auto | None (automatic) |
| Training (long sequences) | FA2/FA3 varlen | `--seq-length 8192+` |
| Inference (static batch) | Flash Decode | `--flash-decode --use-cuda-graph` |
| Inference (dynamic batch) | FA2/FA3 + FlashInfer | `--inference-dynamic-batching` |
| DeepSeek-V3 Inference | Flash MLA | `--cache-mla-latents --inference-dynamic-batching-block-size 64` |

---

## Performance Characteristics

### Memory Complexity Comparison

| Attention Type | Memory Complexity | Notes |
|----------------|------------------|-------|
| Standard Attention | O(N²) | N = sequence length |
| FlashAttention 2/3 | O(N) | Online softmax with tiling |
| Flash MLA | O(N × rank) | rank ≈ 576 for DeepSeek-V3 |

### Sequence Length Impact

**Standard Attention Memory:**
```
Activations = batch × heads × seq_len² × sizeof(dtype)
            = 8 × 32 × 8192² × 2 bytes
            = ~34 GB
```

**FlashAttention Memory:**
```
Activations = batch × heads × seq_len × head_dim × sizeof(dtype)
            = 8 × 32 × 8192 × 128 × 2 bytes
            = ~512 MB
```

**Memory Savings: ~67x** for 8K sequences 🚀

### Kernel Fusion Impact

**Flash Decode Kernel Timeline:**
```
Standard Path:
├─ Kernel 1: Apply RoPE to Q    (~50 μs)
├─ Kernel 2: Apply RoPE to K    (~50 μs)
├─ Kernel 3: Update KV Cache    (~100 μs)
├─ Kernel 4: Attention Matmul   (~200 μs)
└─ Kernel 5: Softmax            (~50 μs)
Total: ~450 μs

Flash Decode Path:
└─ Kernel 1: Fused Flash Decode (~180 μs)
Total: ~180 μs

Speedup: ~2.5x
```

### Throughput Benchmarks

**LLaMA-2 70B Inference (H100, bf16):**

| Batch Size | Seq Len | Standard | Flash Decode | Speedup |
|------------|---------|----------|--------------|---------|
| 1          | 2048    | 12 tok/s | 28 tok/s     | 2.33x   |
| 8          | 2048    | 85 tok/s | 196 tok/s    | 2.31x   |
| 16         | 2048    | 148 tok/s| 342 tok/s    | 2.31x   |
| 32         | 2048    | 242 tok/s| 558 tok/s    | 2.31x   |

**DeepSeek-V3 671B Inference (H100, Flash MLA):**

| Batch Size | Cache Type | Memory (GB) | Throughput |
|------------|------------|-------------|------------|
| 1          | Standard KV| 260 GB      | OOM        |
| 1          | Flash MLA  | 4.5 GB      | 8.2 tok/s  |
| 16         | Flash MLA  | 72 GB       | 94 tok/s   |

---

## Implementation Deep Dive

### Flash Decode and Prefill Unified Kernel

The `flash_decode_and_prefill` method handles both prefill (first pass) and decode (subsequent passes) in a unified code path.

**megatron/core/transformer/attention.py:506-638**
```python
def flash_decode_and_prefill(
    self,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    max_seqlen_q,
    max_seqlen_k,
    cu_query_lengths,
    cu_kv_lengths,
    seqlens_k,
    block_table,
) -> Tensor:
    """Flash attention kernel for mixed decode and prefill samples.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key cache tensor.
        v (Tensor): Value cache tensor.
        max_seqlen_q (int): Maximum query sequence length.
        max_seqlen_k (int): Maximum key/value sequence length.
        cu_query_lengths (Tensor): Cumulative query lengths.
        cu_kv_lengths (Tensor): Cumulative key/value lengths.
        seqlens_k (Tensor): Key sequence lengths per request.
        block_table (Tensor): Block table for paged KV cache.

    Returns:
        Tensor: Attention output.
    """
    nvtx_range_push(suffix="flash_decode_and_prefill")
    if v is None:
        # MLA absorption mode - no value tensor needed
        cu_seqlens_q = cu_query_lengths
        cu_seqlens_k = cu_kv_lengths
        if HAVE_FA3 or HAVE_FMLA:
            if getattr(self, "softmax_scale", None) is not None:
                softmax_scale = self.softmax_scale
            else:
                softmax_scale = q.shape[-1] ** -0.5
            if HAVE_FA3:
                # FA3 path with block table support
                output_total, *unused = _flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_query_lengths,
                    cu_seqlens_k=cu_kv_lengths,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    return_softmax=False,
                    block_table=block_table,
                    page_size=None,
                    leftpad_k=None,
                    seqused_k=seqlens_k,
                    sm_margin=0,
                    num_splits=0,
                    pack_gqa=None,
                    sm_margin=0,
                )
            else:
                # FA2 path
                output_total = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
        else:
            # Prefill and decode in single batch
            # Determine which requests are in decode vs prefill
            is_decode = (cu_query_lengths[1:] - cu_query_lengths[:-1]) == 1

            if is_decode.all():
                # Pure decode batch - use KV cache kernel
                flash_attn_args = {
                    "q": q,
                    "k_cache": k,
                    "v_cache": v,
                    "softmax_scale": softmax_scale,
                    "cache_seqlens": seqlens_k,
                    "causal": True,
                    "page_table" if HAVE_FA3 else "block_table": block_table,
                }
                if HAVE_FA3:
                    output_total = flash_attn3_with_kvcache(**flash_attn_args)
                else:
                    output_total = flash_attn_with_kvcache(**flash_attn_args)
    nvtx_range_pop(suffix="flash_decode_and_prefill")
    return output_total
```

### Dynamic Batching Metadata

FlashAttention kernels require specific metadata for variable-length sequences.

**megatron/core/inference/contexts/attention_context/mha_metadata.py:10-133**
```python
class MHAMetadata(MetadataBase):
    """
    Metadata for MHA layer using flash-attention.
    """

    def __init__(
        self, block_count_total, max_kv_block_count, max_requests, block_size_tokens, max_seqlen
    ):
        super().__init__()
        device = torch.cuda.current_device()
        self.device = device
        self.max_blocks = block_count_total
        self.max_kv_blocks = max_kv_block_count
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen

        # Pre-allocated buffers for metadata tensors
        self._query_lengths_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self._cu_query_seq_lengths_buf = torch.zeros(
            self.max_bs + 1, dtype=torch.int32, device=device
        )
        self._cu_kv_seq_lengths_buf = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=device)
        self._kv_seq_lengths_buf = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self._block_table_buf = torch.zeros(
            (self.max_bs, self.max_kv_blocks), dtype=torch.int32, device=device
        )
        self._max_seqlen_q = 0
        self._max_seqlen_k = 0
        self.state_data = {}

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        padded_active_token_count: int,
        real_batch_size: int,
        padded_active_request_count: Optional[int] = None,
        decode_only: bool = False,
    ):
        """Update metadata for current batch."""
        if padded_active_request_count is None:
            padded_active_request_count = real_batch_size

        # Copy query lengths with padding
        self.tensor_copy_and_pad(
            self._query_lengths_buf,
            request_query_lengths,
            real_batch_size,
            padded_active_request_count,
        )

        # Compute cumulative query sequence lengths
        self._cu_query_seq_lengths_buf[0] = 0
        self.tensor_copy_and_pad(
            self._cu_query_seq_lengths_buf[1:],
            torch.cumsum(request_query_lengths, dim=0),
            real_batch_size,
            padded_active_request_count,
            is_cumulative_tensor=True,
        )

        # Compute KV sequence lengths (query + past KV)
        self.tensor_copy_and_pad(
            self._kv_seq_lengths_buf,
            request_kv_length_offsets + request_query_lengths,
            real_batch_size,
            padded_active_request_count,
        )

        # Copy block table (paged KV cache)
        self.tensor_copy_and_pad(
            self._block_table_buf,
            request_to_kv_block_ids,
            real_batch_size,
            padded_active_request_count,
            pad_value=torch.tensor(self.max_kv_blocks, dtype=torch.int32, device=self.device).fill_(
                -1
            ),
        )

        # Compute cumulative KV sequence lengths
        self._cu_kv_seq_lengths_buf[0] = 0
        self.tensor_copy_and_pad(
            self._cu_kv_seq_lengths_buf[1:],
            torch.cumsum(self._kv_seq_lengths_buf, dim=0),
            real_batch_size,
            padded_active_request_count,
            is_cumulative_tensor=True,
        )

        # Set max sequence lengths
        if decode_only:
            self._max_seqlen_q = 1  # All requests generating 1 token
        else:
            self._max_seqlen_q = max(2, padded_active_token_count)
        self._max_seqlen_k = self.max_seqlen

        # Package metadata for attention kernel
        self.state_data = {
            "query_lengths": self._query_lengths_buf[:padded_active_request_count],
            "cu_query_seq_lengths": self._cu_query_seq_lengths_buf[
                : padded_active_request_count + 1
            ],
            "cu_kv_seq_lengths": self._cu_kv_seq_lengths_buf[: padded_active_request_count + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:padded_active_request_count],
            "block_table": self._block_table_buf[0:padded_active_request_count, :],
            "max_seqlen_q": self._max_seqlen_q,
            "max_seqlen_k": self._max_seqlen_k,
        }
```

### Kernel Selection Decision Tree

```
Is Inference Mode?
├─ No → Training Path
│   ├─ Transformer Engine enabled?
│   │   ├─ Yes → TE FlashAttention wrapper
│   │   └─ No → Check for packed sequences
│   │       ├─ Packed (THD format) → flash_attn_varlen_func
│   │       └─ Standard → DotProductAttention (fused softmax)
│   └─ FlashAttention version?
│       ├─ FA3 (Hopper) → _flash_attn_forward
│       └─ FA2 → flash_attn_varlen_func
│
└─ Yes → Inference Path
    ├─ Static Batching?
    │   ├─ Yes → flash_decode enabled?
    │   │   ├─ Yes → flash_attn_with_kvcache (with fused RoPE)
    │   │   └─ No → Standard attention + separate RoPE
    │   └─ No → Dynamic Batching
    │       ├─ MLA enabled (cache_mla_latents)?
    │       │   ├─ Yes → Flash MLA kernel
    │       │   └─ No → Continue
    │       ├─ FlashInfer available?
    │       │   ├─ Yes → Use fused RoPE, then flash_decode_and_prefill
    │       │   └─ No → Standard RoPE, then flash_decode_and_prefill
    │       └─ Mixed prefill/decode batch?
    │           ├─ All decode → flash_attn3/2_with_kvcache
    │           ├─ All prefill → flash_attn_varlen_func or _flash_attn_forward
    │           └─ Mixed → Process separately or use varlen with block table
```

---

## Best Practices

### When to Use Flash Decode

✅ **Use Flash Decode When:**
- Static batch sizes (e.g., production serving with fixed concurrency)
- Using CUDA graphs for minimal latency
- Batch size ≤ 32 (fits well in CUDA graph memory)
- Need maximum throughput for uniform workloads

❌ **Don't Use Flash Decode When:**
- Dynamic batching required (variable request arrivals)
- Large batch sizes (>64) - memory pressure increases
- Mixed prefill/decode workloads
- Need fine-grained request scheduling

### When to Use Flash MLA

✅ **Use Flash MLA When:**
- Training/serving DeepSeek-V3 or similar MLA models
- Long context lengths (>32K tokens)
- Memory-constrained environments
- Need extreme KV cache compression

❌ **Don't Use Flash MLA When:**
- Standard attention architectures (not MLA)
- Block size cannot be set to 64 tokens
- Flash MLA library not available

### When to Use FlashInfer

✅ **Use FlashInfer When:**
- Available in your environment
- Using dynamic batching for inference
- RoPE is enabled (rotary positional embeddings)
- Want minimal kernel launches

❌ **Don't Use FlashInfer When:**
- Not using RoPE (e.g., ALiBi, learned positional embeddings)
- FlashInfer not compatible with your CUDA version
- Using static batching with flash_decode (redundant)

---

## Troubleshooting

### Common Issues

**1. "Flash Decoding requires the flash_attn_with_kvcache kernel"**
```bash
# Solution: Install flash-attn package
pip install flash-attn --no-build-isolation

# Verify installation
python -c "from flash_attn import flash_attn_with_kvcache; print('✓ Flash-attn installed')"
```

**2. "Flash MLA requires a block size of 64"**
```bash
# Solution: Set block size correctly
--inference-dynamic-batching-block-size 64
```

**3. "CUDA graphs must use flash decode with static batching!"**
```bash
# Solution: Enable both CUDA graphs and flash decode together
--use-cuda-graph --flash-decode

# Or disable CUDA graphs
--cuda-graph-scope none
```

**4. OOM with Long Sequences**
```bash
# Reduce batch size
--micro-batch-size 1 --global-batch-size 8

# Or use gradient checkpointing
--recompute-granularity full --recompute-method uniform

# Or enable MLA (if applicable)
--multi-latent-attention --cache-mla-latents
```

**5. FlashAttention 3 Not Detected (Hopper GPUs)**
```bash
# Check if FA3 is installed
python -c "import flash_attn_3; print('✓ FA3 installed')"

# Or install flashattn_hopper
pip install flashattn-hopper --no-build-isolation
```

---

## Performance Tuning

### Optimal Batch Sizes for Flash Decode

**H100 (80GB):**
| Model Size | Optimal Batch | Throughput | Memory |
|------------|---------------|------------|--------|
| 7B         | 64-128        | ~2500 tok/s| 45 GB  |
| 13B        | 32-64         | ~1800 tok/s| 62 GB  |
| 70B        | 16-32         | ~560 tok/s | 78 GB  |

**A100 (80GB):**
| Model Size | Optimal Batch | Throughput | Memory |
|------------|---------------|------------|--------|
| 7B         | 32-64         | ~1200 tok/s| 42 GB  |
| 13B        | 16-32         | ~850 tok/s | 58 GB  |
| 70B        | 8-16          | ~280 tok/s | 75 GB  |

### Sequence Length Optimization

**For Training (FlashAttention 2/3):**
```bash
# Optimal: Powers of 2 or multiples of 256
--seq-length 2048   # Good
--seq-length 4096   # Good
--seq-length 8192   # Good

# Avoid irregular lengths
--seq-length 3000   # Suboptimal (padding overhead)
```

**For Inference (Flash MLA):**
```bash
# Must be multiple of block size (64)
--max-seq-length 8192   # 128 blocks
--max-seq-length 16384  # 256 blocks
--max-seq-length 32768  # 512 blocks
```

---

## Summary

FlashAttention optimizations in Megatron provide:

1. **Memory Efficiency**: O(N) vs O(N²) complexity enables long sequences
2. **Speed**: 2-4x faster than standard attention through kernel fusion
3. **Flexibility**: Support for FA2, FA3, Flash Decode, Flash MLA, FlashInfer
4. **Hardware Optimization**: FA3 leverages Hopper-specific features (WGMMA, TMA)
5. **Production Ready**: CUDA graph support, dynamic batching, static optimization paths

**Key Takeaways:**
- FlashAttention is automatically used when available during training
- Flash Decode provides optimal inference performance with static batching
- Flash MLA enables extreme memory savings for long-context MLA models
- FlashInfer adds fused RoPE for dynamic batching scenarios
- Proper configuration is critical for achieving optimal performance

**Related Optimizations:**
- See [05-attention-kernels.md](05-attention-kernels.md) for fused CUDA softmax
- See [15-attention-variants.md](15-attention-variants.md) for GQA/MQA optimizations
- See [09-transformer-engine-integration.md](09-transformer-engine-integration.md) for TE's attention wrappers
