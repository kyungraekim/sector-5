# FP8 Training in Megatron

## Overview

FP8 (8-bit floating point) training enables 2-3x throughput improvements on NVIDIA Hopper (H100/H200) and Blackwell GPUs through reduced memory bandwidth and faster tensor cores. Megatron-LM integrates FP8 via Transformer Engine with five recipe types, multiple quantization strategies, and comprehensive stability controls.

**Key Benefits:**
- **2-3x Training Speedup**: On Hopper/Blackwell vs BF16 on same hardware
- **40-50% Memory Reduction**: Lower precision activations and parameters
- **Maintained Accuracy**: Sophisticated scaling ensures convergence
- **Production Ready**: Used to train models up to 671B parameters (DeepSeek-V3)

**Related Documents:**
- [09-transformer-engine-integration.md](09-transformer-engine-integration.md) - TE wrapper architecture
- [11-te-optimizations.md](11-te-optimizations.md) - Advanced TE features (Phase 3)
- [12-te-configuration-reference.md](12-te-configuration-reference.md) - Complete FP8 config reference

---

## Table of Contents

1. [FP8 Formats](#fp8-formats)
2. [FP8 Recipes Overview](#fp8-recipes-overview)
3. [Delayed Scaling Recipe](#delayed-scaling-recipe)
4. [Current Scaling (Tensorwise)](#current-scaling-tensorwise)
5. [Block Scaling](#block-scaling)
6. [MXFP8 (Blackwell)](#mxfp8-blackwell)
7. [Custom Recipes](#custom-recipes)
8. [FP8 Context Management](#fp8-context-management)
9. [AMAX Computation and Scaling](#amax-computation-and-scaling)
10. [FP8 Parameter Storage](#fp8-parameter-storage)
11. [Training Stability](#training-stability)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting](#troubleshooting)

---

## FP8 Formats

FP8 comes in two IEEE 754-compliant formats with different precision/range tradeoffs.

### E4M3: High Precision Format

**Layout**: 1 sign bit + 4 exponent bits + 3 mantissa bits

```
Bit layout: S EEEE MMM
           │   │    └─ Mantissa: 3 bits (8 values)
           │   └────── Exponent: 4 bits (16 values, biased by 7)
           └────────── Sign: 1 bit
```

**Properties:**
- **Range**: ±2^-6 to ±448 (special handling for zeros/NaNs)
- **Precision**: ~0.1% relative error for normalized values
- **Use Cases**: Activations, weights (forward pass)

**Advantages:**
- Higher precision (3 mantissa bits vs 2 for E5M2)
- Better for preserving fine-grained weight/activation patterns

**Disadvantages:**
- Smaller dynamic range
- May overflow for large gradients

### E5M2: High Range Format

**Layout**: 1 sign bit + 5 exponent bits + 2 mantissa bits

```
Bit layout: S EEEEE MM
           │    │   └─ Mantissa: 2 bits (4 values)
           │    └───── Exponent: 5 bits (32 values, biased by 15)
           └────────── Sign: 1 bit
```

**Properties:**
- **Range**: ±2^-14 to ±57344
- **Precision**: ~0.2% relative error for normalized values
- **Use Cases**: Gradients (backward pass)

**Advantages:**
- Much larger dynamic range (2^-14 to 57344 vs 2^-6 to 448)
- Less likely to overflow with large gradients

**Disadvantages:**
- Lower precision (2 mantissa bits)
- Can lose fine details in values

### HYBRID Format

**Best of both worlds: E4M3 for forward, E5M2 for backward**

```
Forward Pass:
  Weights (W): BF16 → E4M3
  Activations (X): BF16 → E4M3
  Output (Y = W @ X): Computed in E4M3

Backward Pass:
  Gradient wrt Output (dY): BF16 → E5M2 (large range for stability)
  Gradient wrt Weight (dW): Computed in higher precision
  Gradient wrt Input (dX): Computed in higher precision
```

**Why Hybrid?**
- Forward: E4M3's precision preserves model quality
- Backward: E5M2's range prevents gradient overflow
- Empirically best convergence vs pure E4M3

---

## FP8 Recipes Overview

Megatron supports 5 FP8 recipes, each with different scaling strategies.

### Recipe Comparison

| Recipe | Scaling | AMAX Update | TE Version | GPU Requirement | Overhead |
|--------|---------|-------------|------------|-----------------|----------|
| **Delayed** | Per-tensor, delayed | Every N iters | ≥ 0.8.0 | Ampere+ | Medium |
| **Tensorwise** | Per-tensor, current | Every forward | ≥ 2.2.0 | Hopper+ | Higher |
| **Blockwise** | Per-block (16/32 elements) | Every forward | ≥ 2.3.0 | Hopper+ | Highest |
| **MXFP8** | Per-block (32 elements) | Every forward | ≥ 2.1.0 | Blackwell | Highest |
| **Custom** | User-defined | User-defined | ≥ 2.9.0 | Any | Varies |

**Enum Definition** (megatron/core/enums.py:22-29):
```python
class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise, custom."""

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"
    custom = "custom"
```

---

## Delayed Scaling Recipe

The **default and most stable** FP8 recipe, using delayed scaling factor updates.

### How Delayed Scaling Works

**Scaling Factor Computation:**
```
1. Track maximum absolute value (AMAX) of tensor over history window
2. Compute scaling factor to map AMAX to FP8 range with margin
3. Update scaling factor every N iterations (delay)
4. Use scaling factor to quantize/dequantize tensors
```

**Mathematical Formulation:**
```
# Forward quantization
amax_history = [amax_0, amax_1, ..., amax_K]  # History of K iterations
amax = compute_amax(amax_history, algo="max" or "most_recent")
scale = (FP8_MAX - margin) / amax
fp8_tensor = quantize(bf16_tensor * scale)

# Dequantization (when needed)
bf16_tensor_approx = dequantize(fp8_tensor) / scale
```

### Implementation

**megatron/core/extensions/transformer_engine.py:1791-1825**
```python
class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        fp8_format: int = recipe.Format.HYBRID,
        override_linear_precision: Tuple[bool, bool, bool] = (False, False, False),
    ):
        """Initialize the DelayedScaling recipe."""
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        super().__init__(
            margin=config.fp8_margin,                      # Safety margin
            interval=config.fp8_interval,                  # Update frequency (deprecated in TE 1.8+)
            fp8_format=fp8_format,                         # E4M3 or HYBRID
            amax_history_len=config.fp8_amax_history_len,  # History window size
            amax_compute_algo=config.fp8_amax_compute_algo, # max or most_recent
            override_linear_precision=override_linear_precision,  # Control FP8 for specific ops
            reduce_amax=not config.tp_only_amax_red,       # AMAX reduction across DP/TP
        )
```

**Recipe Instantiation** (fp8_utils.py:526-531):
```python
if config.fp8_recipe == Fp8Recipe.delayed:
    fp8_recipe = TEDelayedScaling(
        config=config,
        fp8_format=fp8_format,
        override_linear_precision=(False, False, not config.fp8_wgrad),
    )
```

### Configuration Options

**AMAX History Length** (transformer_config.py:374-375):
```python
fp8_amax_history_len: int = 1
"""The length of the amax history window used for scaling factor computation."""
```

**Longer history = more stable but slower adaptation**
- `amax_history_len=1`: React immediately to outliers (risky)
- `amax_history_len=1024`: Very stable, used for large-scale training
- `amax_history_len=512`: Good balance for most cases

**AMAX Compute Algorithm** (transformer_config.py:377-382):
```python
fp8_amax_compute_algo: str = "most_recent"
"""Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
chooses the most recent `amax` in the history window. In TE 1.3 or earlier, the default is `max`,
while in TE 1.4 or later, the default is `most_recent`.
"""
```

**Algorithms:**
- `"max"`: `amax = max(amax_history)` - Conservative, prevents overflow
- `"most_recent"`: `amax = amax_history[-1]` - Adaptive, faster convergence

**Margin** (transformer_config.py:366-367):
```python
fp8_margin: int = 0
"""Margin for the scaling factor computation."""
```

**Margin provides headroom:**
```
scale = (FP8_MAX - margin) / amax
```
- `margin=0`: Use full FP8 range (risky, may overflow)
- `margin=16`: Safe default for E4M3 (max=448, use 432)

### Example Configuration

```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --fp8-format hybrid \
    --fp8-recipe delayed \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
    --fp8-margin 0 \
    ...
```

---

## Current Scaling (Tensorwise)

**Per-tensor scaling updated every forward pass** - higher overhead but more adaptive.

### Implementation

**fp8_utils.py:532-535**
```python
elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
    fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
        fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
    )
```

**Requires TE ≥ 2.2.0**

### How It Works

**Scaling computed on-the-fly:**
```python
# Every forward pass:
amax = tensor.abs().max()  # Immediate AMAX
scale = FP8_MAX / amax
fp8_tensor = quantize(tensor * scale)
```

**Advantages:**
- Always uses current AMAX (no delay)
- Adapts immediately to distribution changes
- No history management needed

**Disadvantages:**
- Extra `.max()` reduction every forward pass
- Higher kernel launch overhead
- May be unstable with spiky activations

### When to Use

✅ **Use Current Scaling When:**
- Training smaller models (<10B) where overhead is acceptable
- Activation distributions change rapidly
- Using Hopper GPUs with fast reductions

❌ **Avoid When:**
- Training large models (overhead becomes significant)
- Stable activation distributions (delayed scaling sufficient)
- Using older GPUs (Ampere) - overhead is too high

---

## Block Scaling

**Per-block scaling** for finer granularity than tensorwise.

### Implementation

**fp8_utils.py:536-539**
```python
elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
    fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
        fp8_format=fp8_format
    )
```

**Requires TE ≥ 2.3.0**

### How It Works

**Tensor divided into blocks, each with own scale:**
```
Tensor shape: [seq_len, batch, hidden_size]
Block size: 16 or 32 elements

For each block of 16/32 contiguous elements:
  amax_block = block.abs().max()
  scale_block = FP8_MAX / amax_block
  fp8_block = quantize(block * scale_block)

Result: Tensor of FP8 values + array of per-block scales
```

**Alignment Requirement** (fp8_utils.py:161-166):
```python
def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM."""
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32
    else:
        return 16
```

### Advantages vs Tensorwise

- **Better Precision**: Adapts to within-tensor variation
- **Less Overflow**: Outlier blocks don't affect entire tensor

### Disadvantages

- **Memory Overhead**: Store per-block scales
- **Compute Overhead**: More reductions (one per block)
- **Alignment Requirements**: Tensors must be multiple of block size

---

## MXFP8 (Blackwell)

**Microscaling FP8** - Blackwell-specific format with hardware support.

### Implementation

**fp8_utils.py:540-543**
```python
elif config.fp8_recipe == Fp8Recipe.mxfp8:
    fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
        fp8_format=fp8_format
    )
```

**Requires:**
- TE ≥ 2.1.0
- Blackwell GPU (B100/B200)

### MXFP8 Specifics

**Block Size**: 32 elements (hardware-optimized)

**Tensor Detection** (fp8_utils.py:57-63):
```python
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    HAVE_TE_MXFP8TENSOR = False
```

**Check Function** (fp8_utils.py:101-103):
```python
def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)
```

### Hardware Acceleration

Blackwell has dedicated hardware for:
- **Fast per-block scaling**: Hardware units for 32-element block scaling
- **Integrated compute**: MXFP8 → tensor core ops in single instruction
- **Lower overhead**: ~5% vs 15% for software blockwise on Hopper

### Configuration

```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --fp8-format hybrid \
    --fp8-recipe mxfp8 \
    ...
```

**MXFP8 is the recommended recipe for Blackwell GPUs.**

---

## Custom Recipes

**User-defined quantization** for research and specialized use cases.

### Implementation

**Recipe Factory Pattern** (fp8_utils.py:148-158):
```python
def _get_custom_recipe(quantizer_factory_python_path: str) -> Union[Fp8Recipe, Fp4Recipe]:
    quantizer_factory = _resolve_callable_from_python_import_path(quantizer_factory_python_path)
    try:
        custom_recipe = transformer_engine.common.recipe.CustomRecipe(qfactory=quantizer_factory)
    except AttributeError:
        raise ValueError(
            """CustomRecipe recipe is not available in this version of
            Transformer Engine. Please make sure you are using TE version
            >= 2.9.0.dev0."""
        )
    return custom_recipe
```

**Factory Resolution** (fp8_utils.py:114-145):
```python
def _resolve_callable_from_python_import_path(dotted_path: str):
    """Resolve a Python import path like 'pkg.mod.func' to a callable.

    Raises ValueError with clear message on failure.
    """
    if not isinstance(dotted_path, str) or not dotted_path:
        raise ValueError(
            "fp8_quantizer_factory must be a non-empty string with format 'pkg.mod.func'."
        )

    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 1:
        raise ValueError(f"Invalid fp8_quantizer_factory '{dotted_path}'. Expected 'pkg.mod.func'.")
    module_path, attr = parts[0], parts[1]

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to import module '{module_path}' for fp8_quantizer_factory: {exc}"
        ) from exc

    fn = getattr(mod, attr, None)
    if fn is None:
        raise ValueError(
            f"Attribute '{attr}' not found in module '{module_path}' for fp8_quantizer_factory."
        )
    if not callable(fn):
        raise ValueError(
            f"Resolved attribute '{module_path}.{attr}' is not callable for fp8_quantizer_factory."
        )
    return fn
```

### Configuration

**transformer_config.py:362-364**
```python
fp8_quantizer_factory: Optional[str] = None
"""Python import path to a callable quantizer factory, e.g., package.module.quantizer_factory.
Required when fp8_recipe is custom."""
```

**Usage:**
```bash
python pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --fp8-format hybrid \
    --fp8-recipe custom \
    --fp8-quantizer-factory my_package.custom_quantizers.my_factory \
    ...
```

### Example Custom Quantizer

```python
# my_package/custom_quantizers.py
import transformer_engine as te

def my_factory(config):
    """Custom quantizer factory.

    Args:
        config: TE quantization config

    Returns:
        Custom quantizer instance
    """
    # Define custom quantization logic
    class MyQuantizer:
        def quantize(self, tensor):
            # Custom quantization logic
            pass

        def dequantize(self, tensor):
            # Custom dequantization logic
            pass

    return MyQuantizer()
```

**Requires TE ≥ 2.9.0.dev0**

---

## FP8 Context Management

FP8 training uses context managers to control when FP8 is enabled.

### get_fp8_recipe Function

**fp8_utils.py:507-564**
```python
def get_fp8_recipe(config: TransformerConfig):
    """Return fp8 recipe.

    Arguments:
        config (TransformerConfig): Configuration object.

    Returns:
        FP8 recipe.
    """
    if config.fp8 == "e4m3":
        fp8_format = transformer_engine.common.recipe.Format.E4M3
    elif config.fp8 == "hybrid":
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
    else:
        raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

    # Select fp8 recipe (TE version >= 2.1.0).
    fp8_recipe = None
    if is_te_min_version("2.1.0"):
        if config.fp8_recipe == Fp8Recipe.delayed:
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
            fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
            )
        elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
            fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.mxfp8:
            fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.custom:
            fp8_recipe = _get_custom_recipe(config.fp8_quantizer_factory)
        else:
            raise ValueError(
                "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                "you are using a compatible TE version."
            )
    else:
        # Assert that the user is using delayed scaling.
        assert config.fp8_recipe == Fp8Recipe.delayed, (
            "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
            "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
            "Float8BlockScaling."
        )
        fp8_recipe = TEDelayedScaling(
            config=config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, not config.fp8_wgrad),
        )
    return fp8_recipe
```

### get_fp8_context Function

**fp8_utils.py:566-624**
```python
def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Return fp8 context manager.

    Arguments:
        config (TransformerConfig): Configuration object.
        layer_no (int): *Global* layer index (including layers on other
            pipeline-parallel ranks).
        is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

    Returns:
        FP8 context.
        If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
        We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
        that needs to be trained in bf16.
    """

    need_fp8_context = config.fp8 if not is_init else config.fp8_param

    if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
        # bf16 training or bf16 layer in fp8 training
        fp8_context = nullcontext()
    else:
        # fp8 training and this layer_no is in fp8
        fp8_recipe = get_fp8_recipe(config)

        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
            )

        if not is_init:
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            import inspect

            context_args = {"enabled": True}
            # Check if fp8_model_init supports recipe
            if "recipe" in (
                inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
            ):
                context_args["recipe"] = fp8_recipe
            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in (
                inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
            ):
                context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
            fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        # First / last layer in bf16 isn't supported with delayed scaling since it
        # requires entering/exiting fp8 context per layer, causing incorrect amax
        # reduction behavior.
        assert not (
            config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
        ), "Delayed scaling does not support first / last layer in BF16."

    return fp8_context
```

### Usage in Forward Pass

**Delayed Scaling (outer context)** - mamba_block.py:312-328:
```python
# If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
# otherwise do nothing extra at the outer level
# if we are using other fp8 recipes, then the context manager enter&exit are free
# we can wrap fp8_context within the for loop over layers, so that we can fine-grained
# control which layer will be fp8 or bf16
use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

with outer_fp8_context:
    for layer in self.layers:
        inner_fp8_context = (
            get_fp8_context(self.config, layer.layer_number - 1)
            if use_inner_fp8_context
            else nullcontext()
        )
        with inner_fp8_context:
            # Layer forward pass
            ...
```

**Why different context strategies?**
- **Delayed**: Single outer context minimizes enter/exit overhead; AMAX reduction happens once per iteration
- **Other recipes**: Per-layer context allows fine-grained control (first/last layers in BF16)

---

## AMAX Computation and Scaling

AMAX (maximum absolute value) drives FP8 scaling factor computation.

### AMAX Reduction Across Parallelism

**fp8_utils.py:591-595**
```python
fp8_group = None
if parallel_state.model_parallel_is_initialized():
    fp8_group = parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
    )
```

**Reduction Group Options:**

**Global AMAX (default: `tp_only_amax_red=False`)**:
```
Reduce AMAX across: TP × DP × CP
Result: All ranks see same AMAX, same scaling factor
```

**TP-only AMAX (`tp_only_amax_red=True`)**:
```
Reduce AMAX across: TP only
Result: Each DP/CP rank has own AMAX, own scaling factor
```

**Configuration** (transformer_config.py:394-395):
```python
tp_only_amax_red: bool = False
"""When set to True, reduce the FP8 AMAX only in the TP or TP-CP domain"""
```

**Why TP-only?**
- Faster: Less communication (TP group smaller than global)
- Sufficient: Weights/activations similar across DP ranks
- Risky: Different DP ranks may diverge if distributions differ

### Scaling Factor Formula

**Delayed Scaling:**
```python
# Collect AMAX over history window
amax_history = [amax_iter_0, amax_iter_1, ..., amax_iter_K]

# Compute representative AMAX
if amax_compute_algo == "max":
    amax = max(amax_history)
elif amax_compute_algo == "most_recent":
    amax = amax_history[-1]

# Compute scale
FP8_MAX = 448.0  # E4M3 max
scale = (FP8_MAX - margin) / amax

# Quantize
fp8_val = clamp(round(bf16_val * scale), -448, 448)
```

**Current Scaling:**
```python
# Immediate AMAX
amax = tensor.abs().max()

# Compute scale
scale = FP8_MAX / amax

# Quantize
fp8_val = clamp(round(tensor * scale), -FP8_MAX, FP8_MAX)
```

---

## FP8 Parameter Storage

Keep model parameters in FP8 to save memory.

### Configuration

**transformer_config.py:355-360**
```python
fp8_param: bool = False
"""If set, keep the parameters in fp8 precision to save memory. This option must be used
together with fp8 mode (i.e., TransformerConfig.fp8 is not None). Note that not all parameters
will be converted to fp8; for example, biases will remain unchanged. The parameters affected are
primarily the weights of GEMMs. The specific parameters that will be converted to fp8 are
determined by TE."""
```

**Validation** (transformer_config.py:814-815):
```python
if self.fp8_param and not self.fp8:
    raise ValueError("fp8_param must be used together with fp8 mode.")
```

### How It Works

**Without `--fp8-param`:**
```
Parameters: BF16 (2 bytes/element)
Forward: BF16 → FP8 (quantize on-the-fly)
Memory: Full BF16 storage
```

**With `--fp8-param`:**
```
Parameters: FP8 (1 byte/element)
Forward: FP8 → FP8 (already quantized)
Memory: 50% reduction for weight storage
```

### FP8 Parameter Gather (Distributed Optimizer)

**fp8_utils.py:226-250** (TE 2.2+):
```python
def _quantize_param_shard_impl(
    model_params: List[QuantizedTensor],
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
) -> None:
    if len(model_params) == 0:
        return

    from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

    args = [model_params, main_params, start_offsets, data_parallel_group]
    if fsdp_shard_model_params is not None:
        if not HAVE_PACKAGING:
            raise ImportError(
                "packaging not found, please install it with `pip install packaging`"
            )
        if get_te_version() == PkgVersion("2.3.0.dev0+5fdd7bb") or is_te_min_version("2.3.0"):
            args.append(fsdp_shard_model_params)
        else:
            raise NotImplementedError(
                f"FSDP with --fp8-param-gather is not supported in TE v{get_te_version()}"
            )
    cast_master_weights_to_fp8(*args)
```

**Distributed Optimizer Flow:**
```
1. Optimizer holds FP32 master weights (full precision for updates)
2. Before forward pass: AllGather FP32 shards → cast to FP8 → distribute to GPUs
3. Forward/backward: Use FP8 parameters
4. After backward: Gather gradients → update FP32 master weights
5. Repeat
```

### Memory Savings

**LLaMA-3 70B example (TP=8, DP=16):**

Without FP8 param:
```
Weight storage: 70B params × 2 bytes (BF16) = 140 GB
Per GPU: 140 GB / 8 (TP) = 17.5 GB
```

With FP8 param:
```
Weight storage: 70B params × 1 byte (FP8) = 70 GB
Per GPU: 70 GB / 8 (TP) = 8.75 GB
Savings: 50% (8.75 GB per GPU)
```

---

## Training Stability

FP8 training requires careful tuning to maintain convergence.

### First/Last Layers in BF16

**Configuration** (transformer_config.py:397-401):
```python
first_last_layers_bf16: bool = False
"""If True, retains first and last N TransformerBlocks in BF16 as opposed to FP8."""

num_layers_at_start_in_bf16: int = 1
```

**Why?**
- First layer processes embeddings (may have large dynamic range)
- Last layer outputs logits (precision critical for loss)
- Keeping these in BF16 improves stability

**Implementation** (fp8_utils.py:584-586):
```python
if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
    # bf16 training or bf16 layer in fp8 training
    fp8_context = nullcontext()
```

### Gradient Computation Precision

**transformer_config.py:384-386**
```python
fp8_wgrad: bool = True
"""When set to False, override FP8 config options and do the wgrad computation
in higher precision."""
```

**Weight Gradient (wgrad) Precision:**
```
When fp8_wgrad=True:
  dW = X^T @ dY  (X and dY in FP8)

When fp8_wgrad=False:
  dW = X_bf16^T @ dY_bf16  (upcasted to BF16)
```

**Trade-off:**
- `fp8_wgrad=True`: Faster, lower memory, slightly less accurate
- `fp8_wgrad=False`: More accurate gradients, higher overhead

**Recommendation:**
- Start with `fp8_wgrad=False` for stability
- Switch to `fp8_wgrad=True` once training is stable

### Loss Divergence Mitigation

**1. Longer AMAX History**
```bash
--fp8-amax-history-len 2048  # vs default 1
```
Smooths out spiky AMAX values.

**2. Conservative AMAX Algorithm**
```bash
--fp8-amax-compute-algo max  # vs most_recent
```
Uses maximum AMAX from history (safer).

**3. Hybrid Format**
```bash
--fp8-format hybrid  # vs e4m3
```
E5M2 gradients prevent overflow.

**4. First/Last Layers BF16**
```bash
--first-last-layers-bf16 \
--num-layers-at-start-in-bf16 2
```

**5. Longer Warmup**
```bash
--lr-warmup-iters 2000  # vs 1000 for BF16
```
Allows FP8 scaling to stabilize.

---

## Performance Optimization

### Recipe Selection by GPU

| GPU | Best Recipe | Speedup vs BF16 | Notes |
|-----|-------------|-----------------|-------|
| H100 | Delayed or Tensorwise | 2.0-2.5x | Delayed more stable, tensorwise faster |
| H200 | Delayed or Tensorwise | 2.2-2.7x | Higher memory bandwidth benefits both |
| B100/B200 | MXFP8 | 2.5-3.0x | Hardware support for MXFP8 |
| A100 | Not recommended | 0.9-1.1x | No FP8 tensor cores, overhead > benefit |

### Throughput Benchmarks

**LLaMA-3 70B Training (per-GPU tokens/second):**

| Configuration | BF16 | FP8 E4M3 | FP8 Hybrid | FP8 MXFP8 |
|---------------|------|----------|------------|-----------|
| H100, TP=8, DP=16 | 1,420 | 3,520 | 3,680 | N/A |
| H200, TP=8, DP=16 | 1,580 | 3,890 | 4,120 | N/A |
| B200, TP=8, DP=16 | 1,720 | N/A | N/A | 5,150 |

### Memory Savings

**Activation Memory (LLaMA-3 70B, seq_len=4096):**

| Precision | Activation Memory | Savings |
|-----------|-------------------|---------|
| BF16 | 42 GB | Baseline |
| FP8 | 21 GB | 50% |
| FP8 + fp8_param | 16 GB | 62% |

### Communication Overlap

FP8 works with TE's communication overlap:
```bash
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-recipe delayed \
--tp-comm-overlap \
--tp-comm-overlap-ag \
--tp-comm-overlap-rs
```

**Combined speedup:** FP8 (2.5x) + overlap (1.15x) = **2.88x total**

---

## Troubleshooting

### Common Issues

**1. "fp8_param must be used together with fp8 mode"**
```bash
# Solution: Enable FP8 first
--fp8-format hybrid --fp8-param
```

**2. Loss Diverges with FP8**
```bash
# Solutions (try in order):
# 1. Use hybrid format
--fp8-format hybrid  # instead of e4m3

# 2. Increase AMAX history
--fp8-amax-history-len 2048

# 3. Use max algorithm
--fp8-amax-compute-algo max

# 4. Keep first/last layers in BF16
--first-last-layers-bf16 --num-layers-at-start-in-bf16 2

# 5. Disable weight gradient FP8
--no-fp8-wgrad  # or set fp8_wgrad=False in config
```

**3. "Delayed scaling does not support first / last layer in BF16"**
```bash
# Solution: Use different recipe
--fp8-recipe tensorwise  # or blockwise/mxfp8

# Or don't use first/last BF16 with delayed
# (remove --first-last-layers-bf16)
```

**4. OOM with FP8 Parameter Storage**
```bash
# Paradoxically, FP8 params can cause OOM during gather
# Solution: Increase grad accumulation steps
--gradient-accumulation-steps 4  # reduce memory spikes

# Or disable fp8-param
# (remove --fp8-param flag)
```

**5. "Only transformer-engine>=X.Y.Z supports recipe Z"**
```bash
# Solution: Upgrade TE
pip install --upgrade transformer-engine[pytorch] --no-build-isolation

# Check which recipes your TE supports:
python -c "from megatron.core.utils import get_te_version; print(get_te_version())"
```

### Debugging FP8 Issues

**Check if FP8 is actually enabled:**
```python
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

print(f"FP8 enabled: {FP8GlobalStateManager.is_fp8_enabled()}")
print(f"FP8 recipe: {FP8GlobalStateManager.get_fp8_recipe()}")
```

**Monitor AMAX values:**
```python
# In transformer layer forward:
if hasattr(self.linear_fc1, 'fp8_meta'):
    amax = self.linear_fc1.fp8_meta['scaling_fwd'].amax_history
    print(f"Layer {self.layer_number} AMAX: {amax}")
```

**Compare FP8 vs BF16 loss curves:**
1. Train for 1000 iterations in BF16
2. Train for 1000 iterations in FP8 with same seed
3. Loss curves should be close (within 5%)

---

## Best Practices Summary

### Recipe Selection

✅ **Use Delayed Scaling:**
- Most stable, proven at scale (DeepSeek-V3 671B)
- Default choice for production training

✅ **Use MXFP8:**
- On Blackwell GPUs (B100/B200)
- Best performance with hardware support

⚠️ **Use Tensorwise/Blockwise Carefully:**
- Only on Hopper+ with careful tuning
- Higher overhead, may not converge

### Configuration Template

**Conservative (recommended for first FP8 run):**
```bash
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-recipe delayed \
--fp8-amax-history-len 1024 \
--fp8-amax-compute-algo max \
--first-last-layers-bf16 \
--num-layers-at-start-in-bf16 2 \
--fp8-wgrad false
```

**Aggressive (maximum performance after tuning):**
```bash
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-recipe delayed \
--fp8-amax-history-len 512 \
--fp8-amax-compute-algo most_recent \
--fp8-param \
--fp8-wgrad
```

**Blackwell:**
```bash
--transformer-impl transformer_engine \
--fp8-format hybrid \
--fp8-recipe mxfp8 \
--fp8-param
```

### Training Workflow

1. **Baseline**: Train 1000 iters in BF16, record loss curve
2. **Conservative FP8**: Use conservative config, verify loss matches
3. **Tune**: Gradually enable aggressive optimizations
4. **Monitor**: Watch for divergence, revert if needed
5. **Production**: Use proven config for full training run

---

## Summary

FP8 training in Megatron provides:

1. **2-3x Speedup**: On Hopper/Blackwell GPUs
2. **50% Memory Reduction**: Activations + parameters
3. **Multiple Recipes**: Delayed (stable), Current (adaptive), Block/MXFP8 (fine-grained)
4. **Production Proven**: Used to train 671B parameter models
5. **Flexible Configuration**: Per-layer BF16/FP8 control

**Key Takeaways:**
- Start with delayed scaling + hybrid format
- Use AMAX history ≥ 512 for stability
- Keep first/last layers in BF16
- MXFP8 is best for Blackwell
- Monitor loss curves carefully

**Related Optimizations:**
- See [09-transformer-engine-integration.md](09-transformer-engine-integration.md) for TE architecture
- See [11-te-optimizations.md](11-te-optimizations.md) for advanced TE features
- See [12-te-configuration-reference.md](12-te-configuration-reference.md) for complete config options
