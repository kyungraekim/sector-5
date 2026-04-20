# 02 · The Training Loop

This file walks through `RayPPOTrainer.fit()` one step at a time, citing the actual code. Everything below refers to `verl/trainer/ppo/ray_trainer.py` unless noted.

---

## 0. What fit() claims to do

From the docstring at `:1260`:

```python
def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC
    to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
```

Two things matter in that sentence:

1. **The driver constructs dataflow by RPC.** `self.actor_rollout_wg.update_actor(...)`, `self.async_rollout_manager.generate_sequences(...)`, etc. all return futures backed by `RayWorkerGroup`.
2. **Advantages are computed on the driver.** This is why `compute_advantage` at `core_algos.py:136` runs synchronously in the main process — per-step it only reshapes a batch that already lives in CPU memory on the driver.

---

## 1. Before the loop: setup

```python
# ray_trainer.py:1278-1304
self.global_steps = 0
self._load_checkpoint()                              # resume
self.checkpoint_manager.update_weights(self.global_steps)  # ship resumed weights to rollout

if self.config.trainer.get("val_before_train", True):
    val_metrics = self._validate()
    logger.log(data=val_metrics, step=self.global_steps)

if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
    rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
    rollout_skip.wrap_generate_sequences()           # replay cached rollouts — dev speedup

progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
self.global_steps += 1
```

Note how the resumed checkpoint is pushed to the rollout replicas (`checkpoint_manager.update_weights`) *before* the first validation — verl keeps the two model copies in lockstep even across resumes.

---

## 2. The per-step sequence (annotated)

Inside `for epoch in ...: for batch_dict in self.train_dataloader:` (line 1316). Each section below corresponds to a clearly delimited block in the source. Line numbers are approximate.

### 2.1 Generate — get responses + rollout-side logprobs

```python
# :1337-1356
gen_batch = self._get_gen_batch(batch)
gen_batch.meta_info["global_steps"] = self.global_steps
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
)

with marked_timer("gen", timing_raw, color="red"):
    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
    self.checkpoint_manager.sleep_replicas()   # <-- rollout sleeps immediately after generation
```

Key points:

- `rollout.n` is the number of samples per prompt. For GRPO this is typically 8–16.
- `generate_sequences` is an RPC into `AsyncLLMServerManager` (see `03-rollout-and-weight-sync.md`). It returns token ids, attention masks, and `rollout_log_probs` (the logprobs emitted *by the rollout engine itself during sampling*).
- `sleep_replicas()` (`verl/checkpoint_engine/base.py:399`) is called eagerly — as soon as the driver has the tokens, the rollout GPUs can release weight + KV-cache memory so the training engine can wake up.

### 2.2 REMAX baseline (optional)

```python
# :1359-1386  —  only if adv_estimator == REMAX
gen_baseline_batch = deepcopy(gen_batch)
gen_baseline_batch.meta_info["do_sample"] = False
gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
```

REMAX uses greedy decoding as the variance-reduction baseline; verl re-enters the rollout once per step for this.

### 2.3 Reward extraction

```python
# :1410-1416
if self.use_rm and "rm_scores" not in batch.batch.keys():
    batch_reward = self._compute_reward_colocate(batch)
    batch = batch.union(batch_reward)

reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

Two paths here:

- **Reward-model path** (neural RM colocated with training): `_compute_reward_colocate` dispatches to the RM worker group.
- **Rule-based path** (math / code / search-style tasks): a `RewardManager` decorated with `@register(...)` at `verl/workers/reward_manager/registry.py:24` is invoked by `extract_reward`. The actual scoring function is selected by `data_source` in `verl/utils/reward_score/__init__.py:44`.

`reward_tensor` shape: `(B, T_response)` — per-token rewards, usually zero except at the EOS position.

### 2.4 Recompute old logprobs (or bypass)

```python
# :1422-1464
rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

if bypass_recomputing_logprobs:
    apply_bypass_mode(batch=batch, ...)            # use rollout_log_probs directly
else:
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
        ...
        batch = batch.union(old_log_prob)
```

**Why recompute?** The rollout engine (vLLM/SGLang) uses different CUDA kernels, dtypes, and parallelism from the trainer, so its logprobs drift from what the trainer would compute for the *same* parameters. Recomputing with the training engine gives numerically consistent `old_log_probs` as the stable proximal anchor for PPO. This is called **decoupled mode** (3 policies: π_rollout, π_old, π_θ). The **bypass mode** skips it and pretends `old_log_probs = rollout_log_probs` — cheaper, but the importance-sampling correction term has to absorb the drift.

`_compute_old_log_prob` is defined at `:1163` and ultimately calls `self.actor_rollout_wg.compute_log_prob(batch)` → `ActorRolloutRefWorker.compute_log_prob` at `verl/workers/engine_workers.py:640`, which runs the training engine in inference mode (`self.actor.infer_batch(data)`).

### 2.5 Reference logprobs (optional)

```python
# :1467-1471
if self.use_reference_policy:
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

Uses `self.ref_policy_wg.compute_ref_log_prob` (`engine_workers.py:633`). Produces `ref_log_prob` of shape `(B, T_response)`. Needed whenever `use_kl_in_reward=True` or `actor.use_kl_loss=True`.

### 2.6 Values (critic-based methods)

```python
# :1474-1477
if self.use_critic:
    values = self._compute_values(batch)
    batch = batch.union(values)
```

`_compute_values` (`:1125`) dispatches to `self.critic_wg.compute_values`. Only used when `adv_estimator=gae` (classic PPO). GRPO/RLOO/REMAX/OPO are critic-free.

### 2.7 KL penalty in reward (optional)

```python
# :1479-1494
batch.batch["token_level_scores"] = reward_tensor

if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(
        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
    )
    metrics.update(kl_metrics)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

`apply_kl_penalty` at `core_algos.py:76` subtracts `β * KL(π_θ || π_ref)` from the token-level reward. The `kl_penalty` arg picks the KL estimator kind (`kl`, `abs`, `mse`, `low_var_kl`, `full`, `k3`, `k3+`, `low_var_kl+` — see commit `7e80ab0c` that fixed the `+` suffix parsing). `kl_ctrl` is adaptive or fixed (`core_algos.py:153`).

### 2.8 Advantages

```python
# :1511-1524
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
    num_repeat=self.config.actor_rollout_ref.rollout.n,
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    config=self.config.algorithm,
)
```

`compute_advantage` at `core_algos.py:136` dispatches via the `ADV_ESTIMATOR_REGISTRY` — each estimator is a function decorated with `@register_adv_est(...)`. Supported names live in `AdvantageEstimator` enum (`core_algos.py:88`):

| Name | Uses critic? | Usual use case |
|------|--------------|----------------|
| `gae` | yes | Classic PPO with value function |
| `grpo`, `grpo_vectorized`, `grpo_passk` | no | Group-relative policy optimization (DeepSeek, DAPO) |
| `reinforce_plus_plus` (+ baseline) | no | Simpler than GRPO, works with short responses |
| `rloo`, `rloo_vectorized` | no | Leave-one-out baseline |
| `remax` | no | Greedy baseline (uses the extra rollout from §2.2) |
| `opo`, `gpg`, `optimal_token_baseline`, `tir_*`, `gdpo` | no | Specialty recipes |

Output: populates `batch.batch["advantages"]` and (for GAE) `batch.batch["returns"]`.

### 2.9 Critic update

```python
# :1526-1531
if self.use_critic:
    with marked_timer("update_critic", timing_raw, color="pink"):
        critic_output = self._update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

`_update_critic` at `:1233` calls `self.critic_wg.update_critic(batch_td)`, which on the worker side (`engine_workers.py` analogue for critic role) runs MSE-on-returns. Mini-batch / micro-batch splitting happens inside `TrainingWorker.train_mini_batch` at `engine_workers.py:236`.

### 2.10 Actor update + weight sync

```python
# :1533-1569
if self.config.trainer.critic_warmup > self.global_steps:
    # Still warming up the critic — no actor update, but still wake the rollout for consistency
    self.checkpoint_manager.update_weights(self.global_steps)
else:
    with marked_timer("update_actor", timing_raw, color="red"):
        actor_output = self._update_actor(batch)

    # ... save-checkpoint logic ...

    # push new weights to rollout replicas
    with marked_timer("update_weights", timing_raw, color="red"):
        self.checkpoint_manager.update_weights(self.global_steps)
```

`_update_actor` at `:1191` calls `self.actor_rollout_wg.update_actor(batch_td)` → `ActorRolloutRefWorker.update_actor` at `engine_workers.py:648`:

```python
# engine_workers.py:648
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="red", role="actor_update")
@_with_routing_replay_flag(enabled=True)
def update_actor(self, data: TensorDict) -> TensorDict:
    output = self.actor.train_mini_batch(data=data)
    return output.cpu() if output is not None else None
```

The actual forward/backward lives in `TrainingWorker.train_batch` (`engine_workers.py:327`) → `engine.train_batch(data, loss_function=self.loss_fn)` → into the selected `BaseEngine`. See `04-engines-megatron-fsdp.md` and `05-losses-and-updates.md`.

After the update, `checkpoint_manager.update_weights(global_steps)` ships the new weights to the rollout. See `03-rollout-and-weight-sync.md` for exactly what happens inside that call.

### 2.11 Validation, checkpoint

```python
# :1576-...
if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
    val_metrics = self._validate()
    ...
```

`_validate` internally calls `generate_sequences` again with the test prompts and evaluates the same reward manager.

---

## 3. Why the order matters

A few subtle ordering details that trip people up:

- **`sleep_replicas()` is called right after generation, not after the whole step.** This lets the *training* engine wake up and use the GPUs for the `_compute_old_log_prob` + `_update_actor` passes. If you add async tool calls that the rollout services *after* sleep, you'll get confusing errors — the tool runner must complete or be checkpointed before the sleep.
- **`old_log_probs` must exist before `compute_advantage`.** For GRPO/DAPO this is a hard assert at `:1465`. If you bypass old-logprob computation, you must either be in REMAX mode with `rollout_log_probs` present or set up the rollout-correction config correctly.
- **`update_weights` is called whether or not the actor updated**, because `critic_warmup` still wants the rollout replicas to stay alive and cached.

---

## 4. Where the dataset lives in this picture

The `self.train_dataloader` yields dicts that `DataProto.from_single_dict(batch_dict)` turns into the batch the loop operates on. The dataloader is built in `TaskRunner.run` from:

- `RLHFDataset` (`verl/utils/dataset/rl_dataset.py:71`) — the default.
- Or a **custom dataset class** loaded via `data.custom_cls.{path,name}` (see `get_dataset_class` at `rl_dataset.py:478`, `load_extern_object` machinery).
- Wrapped in a `StatefulDataLoader` (`torchdata`) so that resumption mid-epoch is deterministic.

That is the *only* extension point for data; everything else (prompts, reward signals, tools) layers on top of whatever dict the dataset returns. See `06-extending-datasets-and-envs.md`.

---

## 5. A TL;DR sequence diagram

```
┌────────────┐      generate_sequences      ┌───────────────┐
│  driver    │ ────────────────────────────▶│  rollout WG   │
│ (fit loop) │                              │ (vLLM/SGLang) │
│            │◀──────── tokens + logprobs ──│               │
│            │       sleep_replicas         │               │
│            │ ────────────────────────────▶│               │
│            │                              └───────────────┘
│            │      compute_log_prob        ┌───────────────┐
│            │ ────────────────────────────▶│   actor WG    │
│            │◀──── old_log_probs ──────────│  (train eng)  │
│            │      compute_ref_log_prob    │               │
│            │ ────────────────────────────▶│   ref WG      │
│            │      compute_values          │  critic WG    │
│            │ ────────────────────────────▶│               │
│            │      compute_advantage       │               │
│            │  (on driver, local CPU)      │               │
│            │      update_actor            │               │
│            │ ────────────────────────────▶│   actor WG    │
│            │◀──── metrics ────────────────│               │
│            │      checkpoint_manager      ┌───────────────┐
│            │      .update_weights         │ actor + rollout
│            │ ────────────────────────────▶│ (weight sync)  │
└────────────┘                              └───────────────┘
```

The detailed mechanics of the last step — how Megatron hands tensors to vLLM/SGLang and when each sleeps — is the subject of the next file.
