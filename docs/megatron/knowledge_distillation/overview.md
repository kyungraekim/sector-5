# Knowledge Distillation: Architecture & Concepts

> **Scope**: How Megatron-LM and NVIDIA ModelOpt implement knowledge distillation (KD)
> **Audience**: Engineers working on or extending the distillation pipeline

---

## What Is Knowledge Distillation?

Knowledge distillation transfers learned representations from a large, pre-trained **teacher**
model to a smaller **student** model. Instead of training the student only on hard labels
(ground-truth tokens), the student also learns from the teacher's **soft labels** — the full
probability distribution over the vocabulary at each position. These soft distributions encode
inter-class relationships (e.g., "happy" is more similar to "glad" than to "table") that
one-hot labels discard entirely.

The core training signal is the KL-divergence between teacher and student output distributions,
optionally combined with intermediate hidden-state losses and the standard language-model
cross-entropy loss.

---

## Two-Library Architecture

Distillation in Megatron-LM is a collaboration between two libraries:

```
+--------------------------------------------------+
|  Megatron-LM                                     |
|  - Training loop, data pipeline, parallelism      |
|  - Model construction (GPTModel, MambaModel)      |
|  - Loss masking & reporting                        |
|  - Pipeline-parallel tensor routing                |
+---------------------------+----------------------+
                            |
                    mtd.convert()
                            |
+---------------------------v----------------------+
|  NVIDIA ModelOpt  (modelopt.torch.distill)       |
|  - DistillationModel wrapper                      |
|  - Loss functions (KL, Cosine, MSE, MFT, MGD)    |
|  - Loss balancers (Static, Dynamic)               |
|  - Forward hooks for activation capture            |
|  - Teacher freezing & state-dict management        |
+--------------------------------------------------+
```

**Megatron-LM** owns the training infrastructure: CLI argument parsing, model construction,
the training loop, checkpointing, and all parallelism strategies. It invokes ModelOpt's
distillation API during model building.

**ModelOpt** owns the distillation engine: the `DistillationModel` wrapper, loss function
library, forward hooks for activation capture, and teacher model lifecycle management.

A Megatron-specific plugin (`modelopt.torch.distill.plugins.megatron`) bridges the two,
providing tensor-parallel-aware loss functions, pipeline-parallel patching, and dynamic loss
balancing.

---

## The `DistillationModel` Wrapper

The central abstraction is `DistillationModel`
(`modelopt/torch/distill/distillation_model.py`), a `DynamicModule` subclass that wraps the
student model and holds a reference to the teacher:

```
DistillationModel (inherits student's class via DynamicModule)
├── _teacher_model          # frozen teacher (requires_grad=False)
├── _layers_to_loss         # {(student_layer, teacher_layer): loss_fn}
├── _loss_balancer          # reduces multiple losses to a scalar
├── _loss_modules           # nn.ModuleList of parameterized losses
├── _hook_handles           # forward-hook handles for activation capture
└── _expose_minimal_state_dict  # hide teacher from checkpoints
```

**Key behaviors**:

1. **Forward pass**: Runs teacher under `torch.no_grad()` + `.eval()`, then runs student via
   `super().forward()`. Forward hooks on designated layers capture intermediate activations
   into `_intermediate_output` attributes.

2. **Loss computation**: `compute_kd_loss()` iterates `_layers_to_loss`, reads captured
   activations from each `(student_layer, teacher_layer)` pair, computes per-pair losses,
   optionally applies a reduction function (for loss masking), and delegates to the loss
   balancer.

3. **State dict**: `state_dict()` hides the teacher via `hide_teacher_model()` context
   manager when `expose_minimal_state_dict=True` (the default), so checkpoints contain only
   the student weights.

Source: `modelopt/torch/distill/distillation_model.py`

---

## Loss Functions

### Logits Losses

| Loss Class | Location | Description |
|-----------|----------|-------------|
| `LogitsKLLoss` | `plugins/megatron.py:292-370` | KL-divergence with TP-aware global softmax. Uses `dist_nn.functional.all_reduce` for differentiable denominator reduction. |
| `TopKLogitsKLLoss` | `plugins/megatron.py:373-458` | KL-divergence restricted to teacher's top-K vocabulary entries. Gathers local top-K per TP rank, selects global top-K, computes dense softmax on the reduced set. |
| `LogitsDistillationLoss` | `losses.py:28-70` | Standard (non-TP-aware) KL-divergence with temperature scaling. Used outside Megatron contexts. |
| `MFTLoss` | `losses.py:73-195` | Mini-Finetuning KL-divergence that corrects teacher distributions when argmax disagrees with ground truth. |

### Intermediate Losses

| Loss Class | Location | Description |
|-----------|----------|-------------|
| `HiddenStateCosineLoss` | `plugins/megatron.py:243-289` | Cosine-embedding loss between hidden states. Recommended for LayerNorm outputs which have full hidden dim even under TP. |
| `MSELoss` | `plugins/megatron.py:222-240` | Mean-squared-error loss on hidden states, reduced over hidden dim. |
| `MGDLoss` | `losses.py:198-275` | Masked Generative Distillation for vision models (Conv2d-based). |

All Megatron-specific losses inherit from `BaseLoss` (`plugins/megatron.py:186-219`) which
provides:
- `pre_forward()`: applies optional projection layer, detaches teacher tensor
- `post_forward()`: transposes from `[s, b]` to `[b, s]`, returns flags for TP reduction

---

## Loss Balancers

### `StaticLossBalancer`

**Location**: `modelopt/torch/distill/loss_balancers.py:74-140`

Applies fixed weights to each KD loss component. If weights don't sum to 1.0, the remainder
is applied to the student's original loss:

```
total = sum(kd_loss_i * weight_i) + (1 - sum(weights)) * student_loss
```

### `LogitsAndIntermediatesLossBalancer`

**Location**: `plugins/megatron.py:461-516`

The default balancer for Megatron distillation. Uses dynamic scaling:

1. **Intermediate scaling**: Scales intermediate loss to match logits loss magnitude:
   ```python
   dynamic_scale = logits_loss.detach() / intermediate_loss.detach()
   intermediate_loss_scaled = intermediate_loss * dynamic_scale
   ```

2. **KD-to-LM scaling** (when `skip_lm_loss=False`): Scales combined KD loss to match
   original LM loss magnitude before applying `kd_loss_scale`:
   ```python
   kd_loss *= original_loss.detach() / kd_loss.detach()
   total_loss = original_loss + kd_loss * kd_loss_scale
   ```

All scaling factors use `.detach()` to prevent them from entering the gradient computation.

---

## `ProjectionLayer`

**Location**: `plugins/megatron.py:519-552`

When student and teacher have different hidden dimensions, a linear projection maps student
activations to the teacher's space before computing intermediate losses:

```python
if student_config.hidden_size == teacher_config.hidden_size:
    self._fit = nn.Identity()
else:
    self._fit = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
```

A single `ProjectionLayer` instance is shared across all intermediate layer pairs
(`plugins/megatron.py:139`). The `sequence_parallel` attribute is set on weight and bias
tensors to ensure correct gradient reduction during backward.

---

## Distillation YAML Config

The distillation run is configured via a YAML file passed with `--export-kd-cfg`:

```yaml
# Which submodules produce logits
logit_layers: ["output_layer", "output_layer"]  # [student, teacher]

# Pairs of intermediate layers for hidden-state losses
intermediate_layer_pairs:
  - ["decoder.layers.0.input_layernorm", "decoder.layers.0.input_layernorm"]
  - ["decoder.final_layernorm", "decoder.layers.30.input_layernorm"]
  # Optional third element per pair: loss type ("cosine" or "mse", default: "cosine")
  - ["decoder.layers.5.input_layernorm", "decoder.layers.10.input_layernorm", "mse"]

# Whether to skip standard LM cross-entropy loss
skip_lm_loss: true

# Scale factor for KD loss when combined with LM loss (only used if skip_lm_loss: false)
kd_loss_scale: 10.0

# Temperature for logit KL-divergence (higher = softer distributions)
logit_kl_temperature: 1.0

# If set, use TopKLogitsKLLoss with this K value instead of full-vocab LogitsKLLoss
logit_kl_topk: 1024
```

**Field details**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logit_layers` | `[str, str]` | `["output_layer", "output_layer"]` | Student and teacher submodule names whose outputs are logits |
| `intermediate_layer_pairs` | `list[tuple]` | `[]` | Pairs (or triples) of layer names for intermediate losses |
| `skip_lm_loss` | `bool` | `true` | Skip computing standard cross-entropy loss |
| `kd_loss_scale` | `float` | `1.0` | Multiplier on KD loss before adding to LM loss |
| `logit_kl_temperature` | `float` | `1.0` | Temperature parameter for KL-divergence |
| `logit_kl_topk` | `int \| null` | `null` | Top-K value; if set, uses `TopKLogitsKLLoss` |

The YAML is parsed into a `DistillationConfig` dataclass
(`plugins/megatron.py:51-79`), then `setup_distillation_config()` (`plugins/megatron.py:98-161`)
constructs the actual `criterion` dict and `loss_balancer` from it.

---

## `LayerwiseDistillationModel` Variant

`LayerwiseDistillationModel` (`modelopt/torch/distill/layerwise_distillation_model.py`) is a
variant that injects teacher layer inputs directly into corresponding student layers during the
forward pass. Unlike the standard model which only captures outputs via hooks, this variant
enables layer-by-layer guided training where each student layer receives the teacher's
representation as additional input.

Registered via the `"layerwise_kd"` mode in `mode.py:83-103`, requiring explicit criterion
pairs (no output-only distillation).

---

## Component Relationship Diagram

```
pretrain_gpt.py
  │
  ├─ add_modelopt_args()          ← megatron/post_training/arguments.py
  │
  └─ pretrain()
       │
       └─ get_model()
            │
            ├─ args.modelopt_enabled = True
            │
            └─ modelopt_gpt_mamba_builder()    ← megatron/post_training/model_builder.py
                 │
                 ├─ MCoreGPTModel(config, ...)         Student model
                 │
                 ├─ _load_teacher_model_config()       Read NEMO YAML → Namespace
                 │    └─ core_transformer_config_from_args()
                 │
                 ├─ _load_teacher_model()               Create + load teacher checkpoint
                 │
                 ├─ setup_distillation_config()         ← plugins/megatron.py
                 │    ├─ LogitsKLLoss / TopKLogitsKLLoss
                 │    ├─ HiddenStateCosineLoss / MSELoss
                 │    ├─ ProjectionLayer
                 │    └─ LogitsAndIntermediatesLossBalancer
                 │
                 ├─ mtd.convert(model, mode=[("kd_loss", config)])
                 │    └─ DistillationModel.modify()     ← distillation_model.py
                 │         ├─ teacher.requires_grad_(False)
                 │         ├─ Register forward hooks
                 │         └─ Store criterion + balancer
                 │
                 └─ adjust_distillation_model_for_mcore()  ← plugins/megatron.py
                      ├─ Patch sharded_state_dict
                      ├─ Patch compute_language_model_loss (skip LM loss)
                      ├─ Patch forward (PP tensor concat)
                      └─ Patch set_input_tensor (PP tensor split)
```
