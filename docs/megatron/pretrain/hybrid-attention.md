# Hybrid Attention in Megatron-LM

## Overview

Many modern transformer models use **hybrid attention**, where most layers use computationally efficient **sliding window attention (SWA)** while a subset of layers use **full (global) causal attention**. This design reduces the quadratic cost of attention for long sequences while preserving the model's ability to attend to distant tokens at regular intervals.

Notable models using this pattern:

| Model | Pattern | Window Size |
|-------|---------|-------------|
| Gemma 3 (Google DeepMind) | 1 full attention every 6 layers (5 SWA + 1 full) | 1024 |
| Gemma 2 (Google DeepMind) | Alternating full and SWA (1:1) | 4096 |
| Mistral | All SWA (no full attention layers) | 4096 |
| Mixtral | All SWA | 4096 |

Megatron-LM natively supports hybrid attention via two CLI arguments: `--window-size` and `--window-attn-skip-freq`.

## How to Use

### Basic Sliding Window Attention (All Layers)

To apply sliding window attention to **every** layer:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --window-size 4096,0 \
    ... # other arguments
```

The `--window-size` argument takes a tuple `(left, right)`:
- `left`: number of tokens to the left the attention can see
- `right`: number of tokens to the right (typically `0` for causal models)
- `-1` means infinite (no window boundary on that side)

### Hybrid Attention with Integer Frequency

To create a repeating pattern where every Nth layer uses full attention:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --window-size 1024,0 \
    --window-attn-skip-freq 6 \
    ... # other arguments
```

With `--window-attn-skip-freq 6` and 24 layers, the pattern is:

| Layer | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 | 11 | **12** | 13 | 14 | 15 | 16 | 17 | **18** | 19 | 20 | 21 | 22 | 23 | **24** |
|-------|---|---|---|---|---|-------|---|---|---|----|----|--------|----|----|----|----|----|----|----|----|----|----|----|----|
| Type  | SWA | SWA | SWA | SWA | SWA | **Full** | SWA | SWA | SWA | SWA | SWA | **Full** | SWA | SWA | SWA | SWA | SWA | **Full** | SWA | SWA | SWA | SWA | SWA | **Full** |

This is the **Gemma 3 pattern**: 5 sliding window layers followed by 1 full attention layer, repeating.

For **Gemma 2** (alternating 1:1), use `--window-attn-skip-freq 2` instead:

| Layer | 1 | **2** | 3 | **4** | 5 | **6** | 7 | **8** | 9 | **10** |
|-------|---|-------|---|-------|---|-------|---|-------|---|--------|
| Type  | SWA | **Full** | SWA | **Full** | SWA | **Full** | SWA | **Full** | SWA | **Full** |

The rule is: layer number divisible by N gets full attention; all others get SWA.

### Hybrid Attention with Custom Pattern

For arbitrary per-layer control, pass a Python list expression:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --window-size 1024,0 \
    --window-attn-skip-freq "([1]*5+[0]*1)*4" \
    ... # other arguments
```

In the list pattern:
- `1` = sliding window attention
- `0` = full attention

The expression `([1]*5+[0]*1)*4` evaluates to `[1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0]` (24 layers). This produces the same Gemma 3-style pattern but with explicit control.

**The pattern length must match `--num-layers`.**

More examples:

```bash
# Full attention only at the first and last layers (24 layers total)
--window-attn-skip-freq "[0]+[1]*22+[0]"

# Alternating: SWA, full, SWA, full, ...
--window-attn-skip-freq "([1,0])*12"

# Dense full attention for the first 6 layers, then all SWA
--window-attn-skip-freq "[0]*6+[1]*18"
```

### Complete Examples

**Gemma 3-style** (5 SWA + 1 full, repeating):

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 26 \
    --hidden-size 2304 \
    --num-attention-heads 8 \
    --num-query-groups 4 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --window-size 1024,0 \
    --window-attn-skip-freq 6 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --data-path /path/to/data \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --train-iters 100000 \
    --lr-decay-iters 100000 \
    --lr-warmup-iters 2000 \
    --bf16
```

**Gemma 2-style** (alternating SWA and full):

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 26 \
    --hidden-size 2304 \
    --num-attention-heads 8 \
    --num-query-groups 4 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --window-size 4096,0 \
    --window-attn-skip-freq 2 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --data-path /path/to/data \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --train-iters 100000 \
    --lr-decay-iters 100000 \
    --lr-warmup-iters 2000 \
    --bf16
```

## Implementation Details

### Configuration

Hybrid attention is configured through two fields in `TransformerConfig` (`megatron/core/transformer/transformer_config.py`):

```python
window_size: Optional[Tuple[int, int]] = None
# If not None, enables sliding window attention.
# Format: (left_window, right_window). -1 means infinite.

window_attn_skip_freq: Optional[Union[int, List[int]]] = None
# Controls which layers use full attention vs SWA.
# Integer N: every Nth layer uses full attention.
# List: per-layer pattern (1=SWA, 0=full).
```

These are mapped from CLI arguments in `megatron/training/arguments.py`:
- `--window-size` is parsed by `tuple_type()` (converts `"4096,0"` to `(4096, 0)`)
- `--window-attn-skip-freq` is parsed by `moe_freq_type()` (handles both integer and list expression strings)

### Per-Layer Decision Logic

The function `is_layer_window_attention()` in `megatron/core/transformer/utils.py` determines whether a given layer uses SWA:

```python
def is_layer_window_attention(window_size, window_attn_skip_freq, layer_number):
    # layer_number is 1-indexed
    if not window_size:
        return False                           # No window_size set -> full attention
    if window_attn_skip_freq is None:
        return True                            # window_size set, no skip -> all SWA
    if isinstance(window_attn_skip_freq, int):
        return layer_number % window_attn_skip_freq != 0  # Every Nth layer is full
    if isinstance(window_attn_skip_freq, list):
        return bool(window_attn_skip_freq[layer_number - 1])  # Explicit per-layer
```

### Attention Backends

The per-layer window size is applied at attention construction time in two backends:

**1. Native DotProductAttention** (`megatron/core/transformer/dot_product_attention.py`):

During `__init__`, the layer calls `is_layer_window_attention()`. If the layer should use SWA, the configured `window_size` tuple is passed to `FusedScaleMaskSoftmax`. Otherwise, `window_size=None` is passed (full attention). The mask is generated by `get_sliding_window_causal_mask()` which creates a banded boolean mask from the `(left, right)` window bounds.

**2. Transformer Engine TEDotProductAttention** (`megatron/core/extensions/transformer_engine.py`):

Same `is_layer_window_attention()` check. If SWA, the `window_size` tuple is passed as `window_size=config.window_size` to TE's `DotProductAttention`. **Requires Transformer Engine >= 1.2.0**; an assertion error is raised otherwise.

### Sliding Window Mask Creation

The mask for native attention is created by `get_sliding_window_causal_mask()` in `megatron/core/transformer/utils.py`:

```python
def get_sliding_window_causal_mask(sq, skv, window_size):
    m = torch.ones(sq, skv, dtype=torch.bool, device="cuda")
    mu = torch.triu(m, diagonal=skv - sq - window_size[0])   # left bound
    ml = torch.tril(mu, diagonal=skv - sq + window_size[1])   # right bound
    return ~ml
```

This produces a boolean mask where `True` means "masked out" (cannot attend). The mask combines the standard causal constraint with the sliding window bounds.

### Data Flow Summary

```
CLI args (--window-size, --window-attn-skip-freq)
    |
    v
TransformerConfig (window_size, window_attn_skip_freq)
    |
    v
DotProductAttention.__init__() / TEDotProductAttention.__init__()
    |  calls is_layer_window_attention(config.window_size,
    |         config.window_attn_skip_freq, layer_number)
    v
Per-layer: window_size tuple or None
    |
    v
FusedScaleMaskSoftmax (native) or TE DotProductAttention (TE backend)
    |
    v
Sliding window causal mask applied during forward pass
```

## Key Source Files

| File | What it contains |
|------|-----------------|
| `megatron/core/transformer/transformer_config.py:196-203` | `window_size` and `window_attn_skip_freq` config fields |
| `megatron/core/transformer/utils.py:451-467` | `is_layer_window_attention()` per-layer decision function |
| `megatron/core/transformer/utils.py:38-45` | `get_sliding_window_causal_mask()` mask creation |
| `megatron/core/transformer/dot_product_attention.py:93-98` | Native attention SWA integration |
| `megatron/core/extensions/transformer_engine.py:1262-1270` | TE backend SWA integration |
| `megatron/training/arguments.py:1632-1641` | CLI argument definitions |
| `megatron/training/arguments.py:231-291` | `moe_freq_type()` and `tuple_type()` parsers |
| `tests/unit_tests/transformer/test_spec_customization.py:143-193` | Unit tests for SWA |