# Selective Recomputation (Gradient Checkpointing) Guide

This guide explains Megatron-LM's **selective recomputation** system — a fine-grained
activation checkpointing strategy that reduces memory usage while minimizing extra compute.
It covers how the system works, what modules are available, and how to extend it for custom
modules.

**Reference**: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Available Recompute Modules](#available-recompute-modules)
4. [How It Works: `core_attn` Example](#how-it-works-core_attn-example)
5. [Two Checkpointing Mechanisms](#two-checkpointing-mechanisms)
6. [Example: Adding Selective Recomputation to a Custom Module](#example-adding-selective-recomputation-to-a-custom-module)
7. [Tips and Considerations](#tips-and-considerations)

---

## Overview

During training, PyTorch saves intermediate activations from the forward pass so they are
available for gradient computation in the backward pass. For large models these activations
consume significant GPU memory — often more than the model parameters themselves.

**Full recomputation** (`--recompute-granularity full`) checkpoints entire transformer layers,
discarding all intermediates and recomputing them during backward. This saves the most memory
but adds ~30-40% compute overhead.

**Selective recomputation** (`--recompute-granularity selective`) takes a smarter approach:
only checkpoint specific submodules that are memory-intensive but compute-cheap. The core
attention softmax outputs, for example, have size `O(s² × h)` (quadratic in sequence length)
but are inexpensive to recompute. By selectively checkpointing only these submodules, you
reclaim most of the memory savings of full recomputation while adding minimal compute overhead
(typically <5%).

---

## Configuration

### CLI Flags

```bash
# Enable selective recomputation with default module (core_attn)
--recompute-granularity selective

# Enable selective recomputation with specific modules
--recompute-granularity selective --recompute-modules core_attn layernorm

# Enable selective recomputation with all available modules
--recompute-granularity selective --recompute-modules core_attn moe_act layernorm mla_up_proj mlp moe shared_experts
```

When `--recompute-granularity selective` is set without `--recompute-modules`, the default
is `["core_attn"]`.

### TransformerConfig Fields

The CLI flags map to these `TransformerConfig` dataclass fields
(see `megatron/core/transformer/transformer_config.py`):

```python
@dataclass
class TransformerConfig:
    # ...
    recompute_granularity: Optional[str] = None
    # Must be 'selective' or 'full'. If None, no recomputation.

    recompute_modules: Optional[List[str]] = None
    # Which submodules to recompute when granularity is 'selective'.
    # Choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts"
    # Default: ["core_attn"]

    recompute_method: Optional[str] = None
    # For 'full' granularity only: 'uniform' or 'block'. Must be None for 'selective'.

    recompute_num_layers: Optional[int] = None
    # For 'full' granularity only. Must be None for 'selective'.
```

**Note**: `recompute_method` and `recompute_num_layers` apply only to `full` granularity.
Selective recomputation always applies to all transformer layers.

---

## Available Recompute Modules

| Module | Checkpointing Mechanism | What It Recomputes | Memory Saved | When to Use |
|--------|------------------------|---------------------|--------------|-------------|
| `core_attn` | Normal (`checkpoint`) | Core attention (softmax, dropout, value weighting) | Attention scores: `O(s² × h)` | Default; always beneficial for long sequences |
| `mlp` | Normal (`checkpoint`) | Entire dense MLP submodule (FC1 + activation + FC2) | MLP intermediates: `O(s × 4h)` | When MLP activations dominate memory |
| `moe` | Normal (`checkpoint`) | Entire MoE layer (route + dispatch + compute + combine) | All MoE intermediates | Large MoE models with memory pressure |
| `shared_experts` | Normal (`checkpoint`) | Shared expert MLP in MoE layers | Shared expert intermediates | MoE models with shared experts enabled |
| `layernorm` | Output-discarding (`CheckpointWithoutOutput`) | `input_layernorm` and `pre_mlp_layernorm` | LayerNorm outputs: `O(s × h)` per norm | Incremental savings on top of other modules |
| `moe_act` | Output-discarding (`CheckpointWithoutOutput`) | MoE MLP activation function (e.g., SwiGLU) | Activation intermediates in expert FFN | MoE models; not compatible with FP8/FP4 legacy GroupedMLP |
| `mla_up_proj` | Output-discarding (`CheckpointWithoutOutput`) | MLA up-projection and RoPE application | Projected Q/K/V tensors | DeepSeek-V3 style MLA models |

### Choosing Modules

Start with `core_attn` (the default) — it provides the best memory-to-compute trade-off.
Add more modules as needed based on your memory budget:

- **Moderate savings**: `core_attn` + `layernorm`
- **Aggressive savings**: `core_attn` + `layernorm` + `mlp`
- **MoE models**: `core_attn` + `moe_act` (or `moe` for maximum savings)
- **MLA models**: `core_attn` + `mla_up_proj`

---

## How It Works: `core_attn` Example

The `core_attn` module demonstrates the **normal checkpointing** pattern. Here is the full
flow from configuration to execution:

### Step 1: TransformerBlock Sets the Flag

When `TransformerBlock.__init__` runs, it checks the config and sets a boolean flag
(`megatron/core/transformer/transformer_block.py`):

```python
# transformer_block.py
self.checkpoint_core_attention = (
    self.config.recompute_granularity == 'selective'
    and "core_attn" in self.config.recompute_modules
)
```

### Step 2: Attention Reads the Flag

`Attention.__init__` reads the same config to set its own flag
(`megatron/core/transformer/attention.py`):

```python
# attention.py
self.checkpoint_core_attention = (
    self.config.recompute_granularity == 'selective'
    and "core_attn" in self.config.recompute_modules
)
```

### Step 3: Forward Pass Conditionally Wraps Core Attention

During `Attention.forward`, the core attention computation is wrapped in a checkpoint call
only during training (`megatron/core/transformer/attention.py`):

```python
# attention.py — inside forward()
if self.checkpoint_core_attention and self.training:
    core_attn_out = self._checkpointed_attention_forward(
        query, key, value, attention_mask,
        attn_mask_type=attn_mask_type,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
    )
else:
    core_attn_out = self.core_attention(
        query, key, value, attention_mask, ...
    )
```

### Step 4: The Checkpointed Forward Implementation

`_checkpointed_attention_forward` defines an inner function and passes it to
`tensor_parallel.checkpoint()` (`megatron/core/transformer/attention.py`):

```python
# attention.py
def _checkpointed_attention_forward(self, query, key, value, attention_mask, ...):
    def custom_forward(*inputs):
        query = inputs[0]
        key = inputs[1]
        value = inputs[2]
        attention_mask = inputs[3]
        attn_mask_type = AttnMaskType(inputs[5].item())
        output_ = self.core_attention(
            query, key, value, attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        return output_

    attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
    hidden_states = tensor_parallel.checkpoint(
        custom_forward, False, query, key, value, attention_mask, rotary_pos_emb, attn_mask_type
    )
    return hidden_states
```

Note: `attention_bias` and `packed_seq_params` are captured via closure rather than passed
as tensor arguments, because `checkpoint()` only tracks tensor inputs for recomputation.

### Step 5: What `checkpoint()` Does Under the Hood

`tensor_parallel.checkpoint()` calls `CheckpointFunction.apply()`
(`megatron/core/tensor_parallel/random.py`):

**Forward pass**:
1. Saves the current RNG states (CPU, CUDA, and the CUDA RNG state tracker for TP)
2. Runs the function under `torch.no_grad()` — intermediates are NOT saved
3. Saves only the **inputs** (query, key, value, mask, etc.) for later recomputation

**Backward pass**:
1. Restores the RNG states to exactly what they were during forward (ensures dropout
   reproducibility)
2. Re-runs the forward function with `torch.enable_grad()` on detached inputs
3. Calls `torch.autograd.backward()` on the recomputed outputs to get gradients

```python
# random.py — CheckpointFunction (simplified)
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.rng_states = _get_all_rng_states()
        with torch.no_grad():
            outputs = run_function(*args)
        ctx.save_for_backward(*args)  # Save inputs, NOT intermediates
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        with _fork_rng():
            _set_all_rng_states(*ctx.rng_states)
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)  # Recompute!
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                       for inp in detached_inputs)
        return (None, None) + grads
```

The key insight: by running the forward under `torch.no_grad()`, PyTorch does not record the
computation graph or save intermediate tensors. Only the inputs are saved. During backward,
the forward is re-run with gradient tracking enabled, creating a fresh computation graph for
just that submodule.

---

## Two Checkpointing Mechanisms

### Normal Checkpointing: `tensor_parallel.checkpoint()`

**Used by**: `core_attn`, `mlp`, `moe`, `shared_experts`

This is the standard gradient checkpointing approach:
- Saves inputs, discards intermediates during forward
- Recomputes the entire wrapped function during backward
- The recomputed output tensor is a **new** tensor (the original output is still alive in the
  autograd graph)

**Pattern**:
```python
output = tensor_parallel.checkpoint(function, distribute_saved_activations, *inputs)
```

**When to use**: When the checkpointed function's output is consumed directly by the next
operation and you don't need to manually control when the output memory is freed.

### Output-Discarding Checkpointing: `CheckpointWithoutOutput`

**Used by**: `layernorm`, `moe_act`, `mla_up_proj`

This is an advanced variant that goes one step further — it also discards the **output**
tensor's storage after the forward pass, then restores it via a backward hook before it's
needed for gradient computation.

**How it works**:
1. `checkpoint()` — runs the function, saves inputs and RNG state, returns output
2. `discard_output_and_register_recompute(hook_tensor)` — resizes the output storage to 0
   (freeing memory) and registers a backward hook on `hook_tensor`
3. When the backward pass reaches `hook_tensor`, the hook fires: recomputes the function,
   restores the output tensor's storage in-place

**Pattern**:
```python
# In __init__
self.recompute_foo = (
    self.config.recompute_granularity == 'selective'
    and "foo" in self.config.recompute_modules
)

# In forward
if self.recompute_foo:
    ckpt = tensor_parallel.CheckpointWithoutOutput()
    output = ckpt.checkpoint(self.foo_module, input_tensor)
    # ... use output in next operation, producing downstream_output ...
    ckpt.discard_output_and_register_recompute(downstream_output)
else:
    output = self.foo_module(input_tensor)
```

**When to use**: When the output of the checkpointed function is saved by a downstream module
(e.g., a linear layer saves its input for backward). In this case, the output is stored
twice — once by the checkpoint and once by the downstream module. Output-discarding
checkpointing eliminates this duplication by freeing the checkpoint's copy and recomputing it
only when needed.

**Critical requirement**: The `hook_tensor` passed to `discard_output_and_register_recompute`
must have its gradient computed **before** any backward operation needs the discarded output.
Choose the hook tensor carefully.

### Concrete Example: `layernorm` Recomputation

In `TransformerLayer.forward()` (`megatron/core/transformer/transformer_layer.py`):

```python
# Forward pass
if self.recompute_input_layernorm:
    self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
    input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
        self.input_layernorm, hidden_states
    )
else:
    input_layernorm_output = self.input_layernorm(hidden_states)

# ... input_layernorm_output feeds into self_attention ...
attention_output_with_bias = self.self_attention(input_layernorm_output, ...)

# After attention: discard the layernorm output and recompute when needed
if self.recompute_input_layernorm:
    self.input_layernorm_checkpoint.discard_output_and_register_recompute(
        attention_output_with_bias[0]
    )
```

The layernorm output is freed after it's consumed by attention. During backward, when the
gradient reaches `attention_output_with_bias[0]`, the hook fires and recomputes the layernorm
to provide its output for the attention backward pass.

---

## Example: Adding Selective Recomputation to a Custom Module

This section walks through adding selective recomputation to a hypothetical custom module —
a "GatedActivation" layer with an expensive activation function.

### Step 1: Register the Module Name

Add your module to the `recompute_modules` choices in two places:

**`megatron/core/transformer/transformer_config.py`** — update the docstring:

```python
recompute_modules: Optional[List[str]] = None
"""The submodules to recompute.
choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe",
         "shared_experts", "gated_activation".
...
"""
```

**`megatron/training/arguments.py`** — update the help text:

```python
group.add_argument('--recompute-modules', nargs='*', type=str, default=None,
                   help='The submodules to recompute. '
                   'choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", '
                   '         "mlp", "moe", "shared_experts", "gated_activation". '
                   ...)
```

### Step 2: Read the Config Flag

In your module's `__init__`, read the config to determine if recomputation is enabled:

```python
from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule


class GatedActivation(MegatronModule):
    def __init__(self, config, ...):
        super().__init__(config=config)
        self.gate_proj = ...  # linear projection
        self.activation_fn = ...  # expensive activation (e.g., mixture of activations)

        # Read the selective recomputation config
        self.recompute_activation = (
            self.config.recompute_granularity == 'selective'
            and "gated_activation" in self.config.recompute_modules
        )
```

### Step 3a: Normal Checkpointing Pattern

Use this when you want to checkpoint the entire submodule computation:

```python
class GatedActivation(MegatronModule):
    # ... __init__ as above ...

    def _checkpointed_forward(self, hidden_states):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            x = inputs[0]
            gate = self.gate_proj(x)
            return self.activation_fn(gate) * x

        return tensor_parallel.checkpoint(custom_forward, False, hidden_states)

    def forward(self, hidden_states):
        if self.recompute_activation and self.training:
            return self._checkpointed_forward(hidden_states)
        else:
            gate = self.gate_proj(hidden_states)
            return self.activation_fn(gate) * hidden_states
```

### Step 3b: Output-Discarding Checkpointing Pattern

Use this when the output is saved by the next downstream module (e.g., a linear layer),
causing duplicate storage:

```python
class GatedActivation(MegatronModule):
    # ... __init__ as above ...

    def forward(self, hidden_states):
        if self.recompute_activation and self.training:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            activation_output = self.activation_checkpoint.checkpoint(
                self._compute_activation, hidden_states
            )
        else:
            activation_output = self._compute_activation(hidden_states)

        # The downstream linear layer will save activation_output for its backward pass.
        output = self.downstream_linear(activation_output)

        if self.recompute_activation and self.training:
            # Discard activation_output storage and recompute when output's grad is ready.
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        return output

    def _compute_activation(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        return self.activation_fn(gate) * hidden_states
```

### Step 4: Test

Write a test verifying that:
1. The forward pass produces the same output with and without recomputation
2. Gradients match between checkpointed and non-checkpointed paths
3. Peak memory is reduced when recomputation is enabled

---

## Tips and Considerations

### RNG State Management

The checkpointing functions save and restore CUDA RNG states to ensure operations like dropout
produce identical results during recomputation. This is handled automatically by
`CheckpointFunction` and `CheckpointWithoutOutput` — you do not need to manage RNG states
manually when using these APIs.

If your custom module uses custom random operations (not through `torch.nn.Dropout`), ensure
they use the CUDA RNG tracker from `tensor_parallel.random.get_cuda_rng_tracker()` for
reproducibility across TP ranks.

### FP8 Compatibility

When FP8 or FP4 training is enabled, normal `tensor_parallel.checkpoint()` may not work
correctly because Transformer Engine requires special handling during recomputation. Use
`te_checkpoint` from `megatron.core.extensions.transformer_engine` instead:

```python
if self.config.fp8 or self.config.fp4:
    from megatron.core.extensions.transformer_engine import te_checkpoint
    output = te_checkpoint(
        function, False,
        tensor_parallel.random.get_cuda_rng_tracker,
        tp_group,
        *inputs,
    )
else:
    output = tensor_parallel.checkpoint(function, False, *inputs)
```

For `CheckpointWithoutOutput`, pass `fp8=self.config.fp8` to the constructor:

```python
ckpt = tensor_parallel.CheckpointWithoutOutput(fp8=self.config.fp8)
```

This ensures proper FP8 autocast context during recomputation.

### CUDA Graph Interactions

Selective recomputation can interact with CUDA graph capture. Some combinations are not
supported — for example, `layernorm` recomputation with MoE router CUDA graphs and shared
expert overlap is automatically disabled with a warning. If you are adding recomputation
to a module that participates in CUDA graph capture, verify that the recomputed tensors
are not stored in static CUDA graph buffers that cannot be resized.

See `TransformerLayer.__init__` in `transformer_layer.py` for the compatibility checks
Megatron performs for the `layernorm` module.

### Distributed Saved Activations

The `distribute_saved_activations` parameter in `tensor_parallel.checkpoint()` allows
splitting saved input activations across the model-parallel group. This further reduces
per-GPU memory but adds communication during backward. It is typically used with `full`
granularity rather than `selective`, but the API supports it for both.

### Closures vs. Tensor Arguments

When defining `custom_forward` functions for `tensor_parallel.checkpoint()`, only **tensor**
arguments are tracked for recomputation. Non-tensor arguments (e.g., enum values, boolean
flags, config objects) should be captured via closure or converted to tensors. The `core_attn`
example demonstrates this: `attention_bias` and `packed_seq_params` are captured via closure,
while `attn_mask_type` is converted to a tensor.

### Memory Profiling

To verify the memory impact of your recomputation choices, use PyTorch's memory profiling:

```python
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
peak_memory = torch.cuda.max_memory_allocated()
```

Compare peak memory with and without your recompute module enabled.
