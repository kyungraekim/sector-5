# 05 · Losses & the Model Update

What actually happens inside `train_batch(loss_function=ppo_loss)`. This file traces forward → loss → backward from the driver's `update_actor()` RPC down to gradient step, with code citations.

---

## 1. Call chain — driver to gradient

```
RayPPOTrainer._update_actor(batch)                    # ray_trainer.py:1191
  → self.actor_rollout_wg.update_actor(batch_td)      #              :1224 (RPC)
    → ActorRolloutRefWorker.update_actor              # engine_workers.py:648
      → self.actor.train_mini_batch(data=data)        #              :649
        → TrainingWorker.train_mini_batch             # engine_workers.py:236 — splits into mini-batches
          → TrainingWorker.train_batch                # engine_workers.py:327
            → self.engine.train_batch(                #
                data, loss_function=self.loss_fn      # loss_fn = ppo_loss by default
              )
              → BaseEngine subclass:
                 - engine.train_mode(...)             # switch to training
                 - forward_backward_batch(...)        # actual forward + backward
                 - optimizer.step() / zero_grad()
                 - lr_scheduler.step()
```

The mini-batch → micro-batch split happens in two layers:

- `train_mini_batch` (line 236) splits the full PPO mini-batch by DP rank. Each PPO epoch iterates over N mini-batches of size `ppo_mini_batch_size`.
- Inside `train_batch` (line 327), the engine itself splits each mini-batch into **micro-batches** using `prepare_micro_batches` (`verl/workers/engine/utils.py`) — respecting `ppo_max_token_len_per_gpu` for dynamic batching.

For Megatron, the micro-batches are fed into `get_forward_backward_func()`'s pipeline schedule (1F1B or interleaved), which handles the forward+backward order across pipeline stages.

---

## 2. `ppo_loss` — the canonical loss

File: `verl/workers/utils/losses.py:58`.

```python
def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
```

### Inputs

`model_output` is whatever the engine's forward pass produced. For PPO it must contain:
- `"log_probs"` — log π_θ of each *response* token, shape `(B, T_response)`.
- `"entropy"` — per-token entropy (optional but normally present).

`data` is the `TensorDict` for this micro-batch, carrying:
- `"response_mask"` — 1 for real response tokens, 0 for padding.
- `"old_log_probs"` — log π_old, from §2.4 of `02-training-loop.md`.
- `"advantages"` — from `compute_advantage`, same shape as log_probs.
- `"ref_log_prob"` — only if `use_kl_loss=True`.
- `"rollout_is_weights"` — importance-sampling weights, only in rollout-correction decoupled mode.

### Step 1: policy-gradient loss

```python
policy_loss_fn = get_policy_loss_fn(loss_mode)   # core_algos.py:70
```

`loss_mode` comes from `config.policy_loss.loss_mode`. Valid modes are whatever's been registered via `@register_policy_loss(...)` in `verl/trainer/ppo/core_algos.py` — common ones:

| loss_mode | Formula (simplified) | When |
|-----------|---------------------|------|
| `vanilla` | `mean(-min(r*A, clip(r, 1-ε, 1+ε)*A))` where `r = exp(logp - old_logp)` | Classic PPO (with GAE) |
| `grpo` | Same clip form, but `A` is group-normalized across `n` samples per prompt | GRPO (DeepSeek / DAPO) |
| `dapo` | PPO clip + separate lower/upper ε for stability on long responses | DAPO |
| `reinforce++` | No clip, just `-logp * A` with optional baseline | REINFORCE++ |
| `opo` | Online Policy Optimization variant | OPO recipe |

Each returns `(pg_loss_scalar, pg_metrics_dict)` — metrics include `pg_clipfrac`, `ppo_kl`, `pg_clipfrac_lower`, etc.

### Step 2: entropy regularizer

```python
# losses.py:124-130
if entropy is not None:
    entropy_loss = agg_loss(
        loss_mat=entropy, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    entropy_coeff = config.entropy_coeff
    policy_loss -= entropy_coeff * entropy_loss
    metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
```

Negative coefficient ⇒ we *subtract* entropy to *reward* high-entropy (exploratory) policies.

### Step 3: KL-in-loss (optional)

```python
# losses.py:132-143
if config.use_kl_loss:
    ref_log_prob = data["ref_log_prob"]
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                       loss_agg_mode=config.loss_agg_mode, **config.global_batch_info)
    policy_loss += kl_loss * config.kl_loss_coef
    metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
    metrics["kl_coef"] = config.kl_loss_coef
```

`kl_penalty` at `core_algos.py` picks the KL estimator kind. Options include `kl`, `abs`, `mse`, `low_var_kl`, `full`, `k3`, `k3+`, `low_var_kl+`. Commit `7e80ab0c` fixed a bug where the `+` suffix wasn't stripped, so `k3+` and `low_var_kl+` failed the string match; current `main` handles them correctly.

### Step 4: aggregation

`agg_loss` (from `core_algos.py`) respects `loss_agg_mode`:
- `"token-mean"` — average loss per non-padded token (the usual choice).
- `"seq-mean-token-sum"` — sum tokens within a sequence, average sequences.
- `"seq-mean-token-mean"` — average tokens within a sequence, then average sequences.

The choice matters: long responses dominate under `token-mean`, but are under-weighted under `seq-mean-*`. DAPO uses a custom mode.

### Return

```python
return policy_loss, metrics   # losses.py:145
```

`policy_loss` is a scalar; the engine's backward pass calls `.backward()` on it (with the appropriate grad scaling for micro-batching / PP).

---

## 3. KL-in-reward vs KL-in-loss

verl supports *both* KL penalties, and they are distinct:

| | KL-in-reward | KL-in-loss |
|---|--------------|------------|
| Where | Driver-side, in `apply_kl_penalty` (`core_algos.py:76`) | Inside `ppo_loss` |
| What | Subtracts `β * KL(π_θ ‖ π_ref)` from per-token reward | Adds `kl_loss_coef * KL` to the loss |
| When set | `algorithm.use_kl_in_reward: true` | `actor.use_kl_loss: true` |
| Effect | Changes advantages (fed through GAE/GRPO) | Directly pushes gradients toward π_ref |
| Typical recipe | Classic RLHF (InstructGPT style) | Many GRPO recipes |

They can coexist. For GRPO-style RL on verifiable rewards, most recipes disable KL-in-reward and rely on a small KL-in-loss coefficient (or none at all — DeepSeek-R1's original recipe uses neither).

---

## 4. Advantage estimators — what's in `batch["advantages"]` before `ppo_loss` sees it

`compute_advantage` (`core_algos.py:136`) dispatches by `adv_estimator`:

```python
# core_algos.py:113
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}

def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name."""
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator
```

Every estimator is `@register_adv_est(...)`-decorated. `AdvantageEstimator` enum (`core_algos.py:88`) names them:

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"
```

Comment at `:91-94` notes: **"Extending this enum for new estimators may not be necessary since users can always just call `verl.trainer.ppo.core_algos.register` with string name for a custom advantage estimator."** So adding a new one is just a decorated function in a module verl imports.

### Quick reference

| Estimator | Uses critic? | Baseline | Variance reduction |
|-----------|--------------|----------|--------------------|
| `gae` | yes | V(s_t) | TD(λ) — best with fitted critic |
| `grpo` | no | mean over `n` samples per prompt | Group-normalized (÷ std) |
| `grpo_passk` | no | pass@k indicator | Used in DAPO |
| `rloo` | no | leave-one-out mean | Unbiased but higher variance than GRPO |
| `remax` | no | greedy rollout | Extra forward pass per step |
| `reinforce_plus_plus` | no | none or batch mean | Simple |
| `opo` / `gpg` / `gdpo` | no | specialty | See `docs/algo/opo.md`, `gpg.md`, `dppo.md` |

---

## 5. Shapes cheat-sheet

When debugging a `ppo_loss` call, these are the shapes you should see:

| Tensor | Shape | Dtype | Where produced |
|--------|-------|-------|----------------|
| `log_prob` (from `model_output`) | `(B, T_response)` | bf16 or fp32 | Trainer-side forward pass |
| `old_log_prob` | `(B, T_response)` | fp32 | `_compute_old_log_prob` |
| `ref_log_prob` | `(B, T_response)` | fp32 | `_compute_ref_log_prob` |
| `advantages` | `(B, T_response)` | fp32 | `compute_advantage` |
| `response_mask` | `(B, T_response)` | bool | rollout post-processing |
| `entropy` | `(B, T_response)` | fp32 | Trainer-side forward (optional) |
| `rollout_is_weights` | `(B, T_response)` | fp32 | Rollout-correction (optional) |

If `no_padding_2_padding` + `to_padded_tensor()` didn't align the shapes, you'll see size mismatches right in the `r = log_prob - old_log_prob` line of whichever `policy_loss_fn` you picked.

---

## 6. Value loss (critic)

For completeness — at `losses.py:148+`:

```python
def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss ..."""
    # Computes MSE between model_output["values"] and data["returns"],
    # clipped to [old_values - clip, old_values + clip] if config.cliprange_value > 0.
```

`compute_value_loss` in `core_algos.py` is the underlying helper. Only active when `use_critic=True` (i.e., `adv_estimator=gae`).

---

## 7. Writing a custom loss

Two small steps:

1. **Define a policy-loss function**:

```python
# my_losses.py
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_clipped_reinforce")
def my_loss(*, old_log_prob, log_prob, advantages, response_mask,
            loss_agg_mode, config, rollout_is_weights=None, **kw):
    # your math here
    pg_loss = ...
    metrics = {"my/pg_clipfrac": ...}
    return pg_loss, metrics
```

2. **Turn it on** via config:

```yaml
actor:
  policy_loss:
    loss_mode: my_clipped_reinforce
```

That's it. `ppo_loss` will resolve `loss_mode="my_clipped_reinforce"` through `get_policy_loss_fn` and feed it the same args every other mode gets. No need to touch the trainer or the engines.

Same pattern for a new **advantage estimator** (`@register_adv_est("my_adv")`) and a new **KL estimator** (add a branch in `kl_penalty` in `core_algos.py` — this one isn't registry-based yet).

---

## 8. Where to go next

- `06-extending-datasets-and-envs.md` — register a new *dataset/reward/tool*, which is what decides what ends up in `batch["advantages"]` in the first place.
- `docs/algo/` (upstream) — per-algorithm deep dives (`dapo.md`, `grpo.md`, `opo.md`, `gpg.md`, ...).
- `verl/trainer/ppo/core_algos.py` itself — every estimator and helper lives in one file; it reads top-to-bottom.
