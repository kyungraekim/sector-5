# Selective MLP Checkpointing

This guide explains how to implement per-layer MLP-only gradient checkpointing in Megatron-LM to reduce memory usage while minimizing recomputation overhead.

## Problem Statement

Megatron-LM's current gradient checkpointing options have limitations:

| Mode             | Module Control              | Layer Control                  | Limitation                      |
| ---------------- | --------------------------- | ------------------------------ | ------------------------------- |
| `selective`      | Yes (`--recompute-modules`) | No (all layers)                | Cannot limit to specific layers |
| `full` + `block` | No (entire layer)           | Yes (`--recompute-num-layers`) | Cannot target only MLP          |

Goal: Apply gradient checkpointing to MLP modules only, and only for a subset of layers (e.g., first N layers).

## Solution Overview

Add a new configuration option `recompute_mlp_num_layers` that controls how many layers receive MLP checkpointing when using selective mode with `mlp` in `recompute_modules`.

## Implementation

{% stepper %}
{% step %}
### File 1: megatron/core/transformer/transformer\_config.py

Add the new configuration field after line 352 (after `recompute_modules` definition):

```python
# Around line 353, add:
recompute_mlp_num_layers: Optional[int] = None
"""When using selective checkpointing with 'mlp' in recompute_modules,
only apply MLP recomputation to the first N layers (1-indexed).
If None, MLP recomputation applies to all layers.
Example: recompute_mlp_num_layers=12 applies MLP checkpointing to layers 1-12 only.
"""
```
{% endstep %}

{% step %}
### File 2: megatron/core/transformer/transformer\_layer.py

Modify the MLP recompute logic around lines 443-445. Replace:

```python
if "mlp" in self.config.recompute_modules:
    if not self.is_moe_layer:
        self.recompute_mlp = True
```

With:

```python
if "mlp" in self.config.recompute_modules:
    if not self.is_moe_layer:
        # Apply MLP recompute only to first N layers if specified
        if self.config.recompute_mlp_num_layers is None:
            self.recompute_mlp = True
        else:
            # layer_number is 1-indexed
            self.recompute_mlp = self.layer_number <= self.config.recompute_mlp_num_layers
```
{% endstep %}

{% step %}
### File 3: megatron/training/arguments.py (CLI)

Add CLI argument after line 2217 (after `--recompute-modules`):

```python
group.add_argument('--recompute-mlp-num-layers', type=int, default=None,
                   help='When using selective checkpointing with "mlp" in '
                   '--recompute-modules, only apply MLP recomputation to the '
                   'first N layers. If not set, applies to all layers. '
                   'Example: --recompute-mlp-num-layers 12 applies MLP '
                   'checkpointing to layers 1-12 only.')
```
{% endstep %}

{% step %}
### File 4: megatron/training/arguments.py (validation)

Add validation in `_validate_args()` function (around line 1000+):

```python
# Validate recompute_mlp_num_layers
if args.recompute_mlp_num_layers is not None:
    if args.recompute_granularity != 'selective':
        raise ValueError(
            '--recompute-mlp-num-layers requires --recompute-granularity selective'
        )
    if args.recompute_modules is None or 'mlp' not in args.recompute_modules:
        raise ValueError(
            '--recompute-mlp-num-layers requires "mlp" in --recompute-modules'
        )
    if args.recompute_mlp_num_layers < 1:
        raise ValueError(
            '--recompute-mlp-num-layers must be >= 1'
        )
```
{% endstep %}
{% endstepper %}

## Usage

### Basic Usage

Checkpoint MLP for first 12 layers only:

```bash
python pretrain_gpt.py \
    --recompute-granularity selective \
    --recompute-modules mlp \
    --recompute-mlp-num-layers 12 \
    --num-layers 24 \
    ...
```

### Combined with Attention Checkpointing

Checkpoint core attention for all layers + MLP for first 12 layers:

```bash
python pretrain_gpt.py \
    --recompute-granularity selective \
    --recompute-modules core_attn mlp \
    --recompute-mlp-num-layers 12 \
    --num-layers 24 \
    ...
```

Note: `core_attn` will still apply to all layers. Only `mlp` respects `--recompute-mlp-num-layers`.

### Programmatic Usage (without CLI)

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    num_layers=24,
    hidden_size=4096,
    num_attention_heads=32,
    ffn_hidden_size=16384,
    recompute_granularity='selective',
    recompute_modules=['mlp'],
    recompute_mlp_num_layers=12,  # Only first 12 layers
)
```

## Memory vs Compute Trade-offs

| Layers with MLP Checkpointing | Memory Savings       | Recompute Overhead    |
| ----------------------------- | -------------------- | --------------------- |
| All layers                    | Maximum              | Maximum               |
| First 50% of layers           | \~50% of max savings | \~50% of max overhead |
| First 25% of layers           | \~25% of max savings | \~25% of max overhead |
| 0 (disabled)                  | None                 | None                  |

Recommended starting points:

* Memory-constrained: Start with all layers (`--recompute-mlp-num-layers` not set)
* Balanced: Start with 50% of layers, adjust based on OOM vs throughput
* Compute-constrained: Use fewer layers or disable MLP checkpointing

## Verification

### Check which layers have MLP checkpointing enabled

Add temporary debug logging in `transformer_layer.py`:

```python
if "mlp" in self.config.recompute_modules:
    if not self.is_moe_layer:
        if self.config.recompute_mlp_num_layers is None:
            self.recompute_mlp = True
        else:
            self.recompute_mlp = self.layer_number <= self.config.recompute_mlp_num_layers

        # Debug logging (remove after verification)
        if parallel_state.get_data_parallel_rank() == 0:
            log_single_rank(
                logger,
                logging.INFO,
                f"Layer {self.layer_number}: recompute_mlp={self.recompute_mlp}"
            )
```

### Unit Test

Create `tests/unit_tests/transformer/test_selective_mlp_recompute.py`:

```python
import pytest
from megatron.core.transformer import TransformerConfig, TransformerLayer
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.attention import SelfAttention


class TestSelectiveMLPRecompute:

    def test_recompute_mlp_num_layers(self):
        """Test that recompute_mlp_num_layers limits MLP checkpointing."""
        config = TransformerConfig(
            num_layers=24,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            recompute_granularity='selective',
            recompute_modules=['mlp'],
            recompute_mlp_num_layers=12,
        )

        # Layers 1-12 should have recompute_mlp=True
        for layer_num in range(1, 13):
            layer = TransformerLayer(config=config, layer_number=layer_num)
            assert layer.recompute_mlp is True, f"Layer {layer_num} should recompute MLP"

        # Layers 13-24 should have recompute_mlp=False
        for layer_num in range(13, 25):
            layer = TransformerLayer(config=config, layer_number=layer_num)
            assert layer.recompute_mlp is False, f"Layer {layer_num} should NOT recompute MLP"

    def test_recompute_mlp_all_layers_when_none(self):
        """Test that all layers recompute MLP when num_layers is None."""
        config = TransformerConfig(
            num_layers=24,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
            recompute_granularity='selective',
            recompute_modules=['mlp'],
            recompute_mlp_num_layers=None,  # Default: all layers
        )

        for layer_num in range(1, 25):
            layer = TransformerLayer(config=config, layer_number=layer_num)
            assert layer.recompute_mlp is True, f"Layer {layer_num} should recompute MLP"
```

## Quick Hack (No Config Changes)

For quick testing without modifying the config system, directly edit `transformer_layer.py:443-445`:

```python
if "mlp" in self.config.recompute_modules:
    if not self.is_moe_layer:
        # QUICK HACK: Only checkpoint MLP for first 12 layers
        # TODO: Replace with config-based approach for production
        _RECOMPUTE_MLP_NUM_LAYERS = 12
        self.recompute_mlp = self.layer_number <= _RECOMPUTE_MLP_NUM_LAYERS
```

## Extending to Other Modules

The same pattern can be applied to other selective recompute modules:

```python
# Example: Add per-layer control for core_attn
recompute_core_attn_num_layers: Optional[int] = None

# In attention.py, modify checkpoint_core_attention logic similarly
```

## Related Configuration Options

| Option                       | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `--recompute-granularity`    | `full` or `selective`                              |
| `--recompute-method`         | `uniform` or `block` (only for `full`)             |
| `--recompute-num-layers`     | Layers per chunk/stage (only for `full`)           |
| `--recompute-modules`        | Which modules to checkpoint (`selective` only)     |
| `--recompute-mlp-num-layers` | **NEW**: Limit MLP checkpointing to first N layers |

## References

* Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198
* Megatron-LM TransformerConfig: `megatron/core/transformer/transformer_config.py`
* Gradient Checkpointing Implementation: `megatron/core/tensor_parallel/random.py`
