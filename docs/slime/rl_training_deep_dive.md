# RL Training Deep Dive

This document is a code-level reference for how slime's RL training pipeline works end-to-end:
from launching training, through text generation and rollout, to model updates via
forward/backward propagation. The final section covers how to add new environments and
datasets (using nvidia/Nemotron-Agentic-v1 as a concrete example).

---

## Table of Contents

1. [Overview and Entry Points](#1-overview-and-entry-points)
2. [Text Generation and Rollout Pipeline](#2-text-generation-and-rollout-pipeline)
3. [Megatron Training and SGLang Inference Interaction](#3-megatron-training-and-sglang-inference-interaction)
4. [Forward and Backward Propagation](#4-forward-and-backward-propagation)
5. [The Sample Type and Data Flow](#5-the-sample-type-and-data-flow)
6. [Adding a New Dataset or Environment](#6-adding-a-new-dataset-or-environment)
7. [Extension Points Reference](#7-extension-points-reference)

---

## 1. Overview and Entry Points

### 1.1 Two Entry Points

Slime provides two training scripts:

| Script | Mode | Key Difference |
|--------|------|----------------|
| `train.py` | Synchronous | Rollout completes before training begins. Supports `--colocate`. |
| `train_async.py` | Async prefetch | Overlaps next rollout with current training step. Does not support `--colocate`. |

Both call `parse_args()` from `slime/utils/arguments.py`, then enter a `train(args)` function.

### 1.2 Initialization Order (Load-Bearing)

The initialization order in both entry points is:

```
1. create_placement_groups(args)      # Allocate GPU bundles via Ray
2. create_rollout_manager(args, pg)   # Start SGLang inference engines
3. create_training_models(args, pgs)  # Initialize Megatron actor (and optional critic)
4. actor_model.update_weights()       # Sync initial weights to rollout engines
```

**Why this order matters:** The rollout manager must exist before training actors are created
because `update_weights()` calls `rollout_manager.get_updatable_engines_and_lock()`. Swapping
the order produces a vague `AttributeError` at weight-update time, not at init time.

These functions live in `slime/ray/placement_group.py`:
- `create_placement_groups()` (line 79) -- creates isolated GPU placement groups for actor, critic, and rollout
- `create_rollout_manager()` (line 183) -- instantiates the `RolloutManager` Ray actor
- `create_training_models()` (line 133) -- creates `RayTrainGroup` objects wrapping `MegatronTrainRayActor` instances

### 1.3 The Main Training Loop (Synchronous)

```python
# train.py, lines 73-103
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # Phase 1: Generate rollout data (text generation + reward computation)
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

    # Phase 2: Train on the generated data
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

    # Phase 3: (Optional) Save checkpoint
    if should_run_periodic_action(rollout_id, args.save_interval, ...):
        actor_model.save_model(rollout_id, ...)

    # Phase 4: Sync updated weights to rollout engines
    actor_model.update_weights()

    # Phase 5: (Optional) Evaluate
    if should_run_periodic_action(rollout_id, args.eval_interval, ...):
        ray.get(rollout_manager.eval.remote(rollout_id))
```

In **colocate mode** (`--colocate`), training and inference share GPUs. The loop
includes explicit offload/onload calls:

```
offload rollout -> train -> onload weights -> update weights -> onload KV
```

Flipping onload-weights and update-weights causes CUDA OOM. Flipping update-weights
and onload-KV overwrites the KV cache.

### 1.4 Async Training Loop

```python
# train_async.py, lines 36-76
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_curr_ref = ray.get(rollout_data_next_future)       # Wait for current

    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)  # Start next early

    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))  # Train on current

    if (rollout_id + 1) % args.update_weights_interval == 0:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)   # Must sync before weight update
        rollout_data_next_future = None
        actor_model.update_weights()
```

The key difference is that rollout `N+1` runs concurrently with training on rollout `N`.
Weight updates happen every `--update-weights-interval` steps (default 1) and must drain
the pending generation first to avoid updating weights mid-inference.

---

## 2. Text Generation and Rollout Pipeline

### 2.1 Architecture Overview

```
                    RolloutManager (Ray Actor)
                    slime/ray/rollout.py
                           |
            +--------------+--------------+
            |                             |
     DataSource                    SGLang Engines
  slime/rollout/data_source.py    slime/backends/sglang_utils/
            |                             |
   Prompt Samples               HTTP /generate requests
            |                             |
            +-----> generate_rollout() <--+
                    slime/rollout/sglang_rollout.py
                           |
                    +------+------+
                    |             |
              generate()    async_rm()
           (text generation)  (reward)
                    |             |
                    +------+------+
                           |
                  RolloutFnTrainOutput
                   (list of Samples)
```

### 2.2 RolloutManager

Defined in `slime/ray/rollout.py` (line 350+), `RolloutManager` is a `@ray.remote` actor
that coordinates the entire rollout lifecycle:

```python
@ray.remote
class RolloutManager:
    def __init__(self, args, pg):
        self.data_source = data_source_cls(args)                          # Load prompt dataset
        self.generate_rollout = load_function(args.rollout_function_path)  # Default: sglang_rollout.generate_rollout
        self.servers = start_rollout_servers(args, pg)                     # Launch SGLang engine processes
```

**Key remote methods:**

| Method | Purpose |
|--------|---------|
| `generate(rollout_id)` | Generate rollout data for training |
| `eval(rollout_id)` | Run evaluation on configured eval datasets |
| `get_updatable_engines_and_lock()` | Return engine refs for weight sync (acquires lock) |
| `offload()` / `onload_weights()` / `onload_kv()` | GPU memory management for colocate mode |

The `generate()` method (line 479) does three things:
1. **Generate rollout data** via `_get_rollout_data()` which calls the user-configurable rollout function
2. **Post-process rewards** via `_convert_samples_to_train_data()` (normalization, conversion to tensors)
3. **Split by data-parallel rank** via `_split_train_data_by_dp()`

### 2.3 Default Rollout Function: `sglang_rollout.generate_rollout`

Located at `slime/rollout/sglang_rollout.py`, this is the default `--rollout-function-path`.

**Top-level flow** (`generate_rollout_async()`, line 389):

```python
async def generate_rollout_async(args, rollout_id, data_source):
    state = GenerateState(args)                           # Process-wide singleton with semaphore

    while collected < target_data_size:
        samples = data_source.get_samples(batch_size)     # Fetch prompt groups from dataset
        state.submit_generate_tasks(samples)               # Create asyncio tasks per group
        # ... wait for completions, apply dynamic filters
        data.extend(completed_groups)

    return RolloutFnTrainOutput(samples=data)
```

**Per-group generation** (`generate_and_rm_group()`, line 310):

For each group of `n_samples_per_prompt` samples sharing the same prompt:
1. Assign unique `session_id` per sample (for consistent hashing routing)
2. Launch concurrent `generate_and_rm()` calls for all samples in the group
3. If `--group-rm` is set, call `batched_async_rm()` on the entire group after generation

**Per-sample generation** (`generate_and_rm()`, line 240):

```python
async def generate_and_rm(args, sample, sampling_params, evaluation=False):
    # 1. Check for custom generation function
    custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path
    if custom_func_path is not None:
        sample = await custom_generate_func(args, sample, sampling_params)
    else:
        sample = await generate(args, sample, sampling_params)

    # 2. Compute reward
    if sample.reward is None:
        sample.reward = await async_rm(args, sample)

    return sample
```

**The `generate()` function** (line 145) sends an HTTP request to the SGLang router:

```python
async def generate(args, sample, sampling_params):
    prompt_ids = _prepare_prompt_ids(sample, tokenizer, processor)

    payload = {"sampling_params": sampling_params, "return_logprob": True, ...}
    payload["input_ids"] = prompt_ids   # or "text" + "image_data" for multimodal

    output = await post(url, payload)   # HTTP POST to SGLang router

    # Parse response
    new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
    new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

    sample.tokens = sample.tokens + new_response_tokens
    sample.response += output["text"]
    sample.rollout_log_probs += new_response_log_probs
    sample.update_from_meta_info(args, output["meta_info"])  # Set status, spec info, etc.

    return sample
```

### 2.4 SGLang Inference Engines

Defined in `slime/backends/sglang_utils/sglang_engine.py` (line 103), each `SGLangEngine`
is a Ray actor that runs an SGLang HTTP server process:

```
SGLangEngine (Ray Actor)
    |
    +-- init() -> _init_normal()
    |       |-- Launch HTTP server subprocess on assigned port
    |       |-- Register with shared router
    |       |-- Wait for /health endpoint to be ready
    |
    +-- Key HTTP endpoints:
        /generate              -- Text generation
        /update_weights_from_tensor  -- Weight sync from training
        /pause_generation      -- Pause for weight update
        /continue_generation   -- Resume after weight update
        /flush_cache           -- Clear KV cache
```

Multiple engines are grouped into a `ServerGroup` (line 38), and server groups are wrapped
by `RolloutServer` (line 211) which handles lifecycle management, fault recovery, and
disaggregated prefill/decode deployments.

### 2.5 Reward Computation

Rewards are computed via `async_rm()` in `slime/rollout/rm_hub/__init__.py` (line 55):

```python
async def async_rm(args, sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    # Built-in reward models:
    match args.rm_type:
        case "remote_rm":    return await remote_rm(args, sample)        # HTTP POST to external RM
        case "deepscaler":   return deepscaler_reward(sample)            # Rule-based math reward
        case "math":         return math_reward(sample)                  # Math verification
        case "gpqa":         return gpqa_reward(sample)                  # GPQA-specific
        case "f1":           return f1_reward(sample)                    # F1 score
        case ...
```

**Reward post-processing** happens in `RolloutManager._post_process_rewards()` (line 656):
- For GRPO/GSPO: group normalization (subtract group mean, optionally divide by group std)
- Custom post-processing via `--custom-reward-post-process-path`

### 2.6 Dynamic Sampling Filters

After generation and reward computation, groups can be filtered via `--dynamic-sampling-filter-path`.
The default filter `check_reward_nonzero_std` (in `slime/rollout/filter_hub/dynamic_sampling_filters.py`)
drops groups where all samples have the same reward (zero variance):

```python
def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(keep=keep, reason=None if keep else f"zero_std_{round(rewards[0], 1)}")
```

---

## 3. Megatron Training and SGLang Inference Interaction

### 3.1 GPU Allocation

Slime allocates separate GPU placement groups for training and inference:

```
+-------------------------------------------+
|  Ray Cluster                              |
|                                           |
|  +------------------+  +---------------+  |
|  | Actor PG         |  | Rollout PG    |  |
|  | (Megatron TP/PP) |  | (SGLang TP)   |  |
|  | GPU 0,1,2,3      |  | GPU 4,5,6,7   |  |
|  +------------------+  +---------------+  |
|                                           |
|  +------------------+                     |
|  | Critic PG (opt)  |                     |
|  +------------------+                     |
+-------------------------------------------+
```

In **colocate mode** (`--colocate`), training and inference share the same GPUs. The system
alternates between them using offload/onload:

```
Rollout (GPU) -> Offload rollout (CPU) -> Train (GPU) -> Onload weights (GPU) -> 
Update weights -> Onload KV cache (GPU) -> Rollout (GPU) -> ...
```

### 3.2 Training Actor Hierarchy

```
RayTrainGroup                          # slime/ray/actor_group.py
    |-- _actor_handlers: list[Ray ActorHandle]
    |       |-- MegatronTrainRayActor  # slime/backends/megatron_utils/actor.py
    |               |-- model          # Megatron GPTModel (DDP-wrapped)
    |               |-- optimizer      # MegatronOptimizer
    |               |-- weights_backuper  # TensorBackuper for model snapshots
    |               |-- weight_updater    # UpdateWeightFromDistributed
    |
    |-- async_init()    -> ray.get() -> model/optimizer initialized
    |-- async_train()   -> ObjectRef (call ray.get() to block)
    |-- update_weights() -> sync weights to rollout engines
    |-- save_model()    -> checkpoint to disk
```

`RayTrainGroup` (line 10 in `slime/ray/actor_group.py`) creates one `MegatronTrainRayActor`
per rank (GPU) and manages distributed training across all ranks.

### 3.3 Weight Synchronization

After training completes, the updated actor weights must be synced to the SGLang rollout
engines. This is handled by `MegatronTrainRayActor.update_weights()` (line 543 in
`slime/backends/megatron_utils/actor.py`):

```python
def update_weights(self):
    # 1. Acquire lock on rollout engines
    rollout_engines, rollout_engine_lock, num_new_engines, ... = ray.get(
        self.rollout_manager.get_updatable_engines_and_lock.remote()
    )

    # 2. Connect to new engines if any (NCCL group setup)
    if num_new_engines > 0:
        self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock, ...)

    # 3. Perform the weight update
    self.weight_updater.update_weights()
```

The actual transfer is in `UpdateWeightFromDistributed.update_weights()` (line 82 in
`slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`):

```python
def update_weights(self):
    self.weight_version += 1

    # Pause inference on all engines
    ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

    # Transfer non-expert parameters
    for name, param in named_params_and_buffers(self.model):
        if ".experts." in name: continue
        self._update_weight_from_distributed(name, param, ...)
        # Internally: AllGather across TP -> NCCL broadcast to rollout engines

    # Transfer expert parameters (separate NCCL channel for MoE)
    for name, param in named_params_and_buffers(self.model):
        if ".experts." not in name: continue
        self._update_expert_weight_from_distributed(name, param, ...)

    # Resume inference
    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

**Weight sync pattern for a single parameter:**
1. Training rank (DP=0, TP=0, each PP rank) holds a shard of the parameter
2. AllGather across the TP group to reconstruct the full parameter on rank 0
3. NCCL broadcast from training rank 0 to all rollout engine processes
4. Rollout engines update their in-memory model weights

### 3.4 TensorBackuper and Model Tags

The `TensorBackuper` (in `slime/backends/megatron_utils/actor.py`) maintains CPU snapshots
of model weights under named tags:

| Tag | Purpose |
|-----|---------|
| `"actor"` | Current training model weights (backed up after each training step) |
| `"ref"` | Reference model for KL divergence computation |
| `"teacher"` | Teacher model for on-policy distillation (OPD) |
| `"old_actor"` | Previous actor version (for `--keep-old-actor` multi-version sampling) |
| `"rollout_actor"` | Copy of weights last sent to rollout engines |

When computing log-probs, the actor swaps between these snapshots via `_switch_model()`,
which copies the tagged CPU weights back to GPU. This is cheaper than maintaining
separate GPU models.

---

## 4. Forward and Backward Propagation

### 4.1 The Actor Training Step (`train_actor()`)

The full actor training step in `MegatronTrainRayActor.train_actor()` (line 406 in
`slime/backends/megatron_utils/actor.py`) proceeds in these stages:

```
Stage 1: Data Preparation
    get_data_iterator(args, model, rollout_data)
    -> DataIterator with micro-batch schedule

Stage 2: Compute Reference Log-Probs (forward-only, no gradient)
    _switch_model("ref")
    compute_log_prob(data_iterator, ..., store_prefix="ref_")
    -> rollout_data["ref_log_probs"]

Stage 3: Compute Teacher Log-Probs (forward-only, optional OPD)
    _switch_model("teacher")
    compute_log_prob(data_iterator, ..., store_prefix="teacher_")
    -> rollout_data["teacher_log_probs"]

Stage 4: Compute Current Actor Log-Probs (forward-only)
    _switch_model("actor")
    compute_log_prob(data_iterator, ..., store_prefix="")
    -> rollout_data["log_probs"]

Stage 5: Compute Advantages and Returns
    compute_advantages_and_returns(args, rollout_data)
    -> rollout_data["advantages"], rollout_data["returns"]

Stage 6: Training Forward-Backward
    train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches)
    -> gradient descent steps

Stage 7: Backup Updated Weights
    weights_backuper.backup("actor")
    -> save new weights to CPU
```

### 4.2 Forward-Only Pass (Reference/Teacher Models)

`forward_only()` in `slime/backends/megatron_utils/model.py` (line 152) runs the model in
eval mode without gradient tracking:

```python
@torch.no_grad()
def forward_only(f, args, model, data_iterator, num_microbatches, store_prefix=""):
    for model_module in model:
        model_module.eval()                  # Disable dropout

    for step_id in range(num_steps_per_rollout):
        forward_backward_func(
            forward_step_func=forward_step,  # Runs model(**forward_kwargs)
            model=model,
            forward_only=True,               # No backward pass
        )

    for model_module in model:
        model_module.train()                 # Re-enable dropout
```

The callback `f` is typically `get_log_probs_and_entropy` (line 382 in `loss.py`), which
extracts per-token log-probabilities from the model logits:

```python
def get_log_probs_and_entropy(logits, *, args, unconcat_tokens, total_lengths, response_lengths, ...):
    # logits shape: [1, T, V] where T = sum of all sequence lengths, V = vocab size
    #
    # For each sample, extract the response portion of logits,
    # compute log P(token_t | token_<t) and entropy H(P(. | token_<t))
    for logits_chunk, tokens_chunk in get_responses(logits, ...):
        log_prob, entropy = calculate_log_probs_and_entropy(logits_chunk, tokens_chunk, ...)
        # log_prob shape: [response_length] -- one value per generated token
    return {}, {"log_probs": log_probs_list, "entropy": entropy_list}
```

`calculate_log_probs_and_entropy` (in `slime/utils/ppo_utils.py`) uses Megatron's
`fused_vocab_parallel_cross_entropy` for efficient, TP-aware log-probability computation.

### 4.3 Advantage Computation

`compute_advantages_and_returns()` (line 572 in `loss.py`) computes advantage estimates
in-place on `rollout_data`. The supported estimators are:

#### GRPO (Group Relative Policy Optimization)

```python
# No value function baseline. Advantages come directly from shaped returns.
kl = [compute_approx_kl(log_probs[i], ref_log_probs[i]) for i in range(N)]
returns = get_grpo_returns(rewards, kl)
# For each sample i: return_i = reward_i - kl_coef * sum(kl_i)
advantages = returns
```

With GRPO, rewards are already group-normalized in `_post_process_rewards()` (subtract
group mean, optionally divide by group std), so the advantages reflect relative
performance within each prompt group.

#### PPO (Proximal Policy Optimization)

```python
# Uses a learned value function as baseline.
# Per-token rewards include KL penalty:
for i in range(N):
    token_rewards[i] = -kl_coef * kl[i]
    token_rewards[i][-1] += scalar_reward[i]     # Add sparse reward to last token

advantages, returns = get_advantages_and_returns_batch(
    total_lengths, response_lengths, values, token_rewards, gamma, lambd
)
# GAE(gamma, lambda): A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
#   where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

#### REINFORCE++ and REINFORCE++ Baseline

Token-level reward redistribution variants with optional moving-average baselines.

#### Advantage Normalization

When `--normalize-advantages` is set, advantages are whitened (zero mean, unit variance)
across the entire data-parallel group using `distributed_masked_whiten()`:

```python
# Compute mean and variance across all DP ranks, respecting loss masks
advantages = distributed_masked_whiten(advantages, loss_masks, dp_group)
```

### 4.4 Training Forward-Backward (`train_one_step`)

`train_one_step()` (line 301 in `model.py`) executes a single gradient accumulation step:

```python
def train_one_step(args, rollout_id, step_id, data_iterator, model, optimizer, opt_param_scheduler, num_microbatches):
    # 1. Zero gradients
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # 2. Define forward step (called by Megatron's pipeline engine)
    def forward_step(data_iterator, model):
        batch = get_batch(data_iterator, [...])  # Get micro-batch
        output_tensor = model(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            packed_seq_params=batch["packed_seq_params"],
            loss_mask=batch["full_loss_masks"],
        )
        return output_tensor, partial(loss_function, args, batch, num_microbatches)

    # 3. Forward + backward via Megatron's pipeline engine
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        forward_only=False,    # Enable backward pass
    )

    # 4. Optimizer step
    if valid_step:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        opt_param_scheduler.step(increment=args.global_batch_size)

    # 5. Release gradients
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
```

Megatron's `get_forward_backward_func()` returns the appropriate pipeline-parallel schedule
(1F1B, interleaved, etc.) which handles gradient accumulation across `num_microbatches`.

### 4.5 Policy Loss Function

`loss_function()` (line 1115 in `loss.py`) dispatches to the configured loss type. For RL
training, the default is `"policy_loss"` which calls `policy_loss_function()` (line 785):

```python
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    # 1. Compute current log-probs from model output
    _, log_probs_and_entropy = get_log_probs_and_entropy(logits, ...)
    log_probs = log_probs_and_entropy["log_probs"]

    # 2. Compute KL divergence between current and old policy
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    ppo_kl = old_log_probs - log_probs   # log(pi_old / pi_current)

    # 3. Clipped policy gradient loss (PPO-style)
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
```

The core PPO clipping logic in `compute_policy_loss()` (`slime/utils/ppo_utils.py`, line 126):

```python
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()                                      # pi_current / pi_old
    pg_losses1 = -ratio * advantages                              # Unclipped
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages  # Clipped
    pg_losses = torch.maximum(pg_losses1, pg_losses2)             # Pessimistic bound

    # Optional dual-clip for negative advantages (DAPO-style)
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        pg_losses = torch.where(advantages < 0, torch.min(pg_losses3, pg_losses), pg_losses)

    return pg_losses, clipfrac
```

The final loss combines:
- **Policy gradient loss** (clipped PPO objective)
- **Entropy bonus** (`-args.entropy_coef * entropy`) to encourage exploration
- **KL penalty** (`args.kl_loss_coef * kl_loss`) to stay close to reference model
- **Optional TIS** (Truncated Importance Sampling) correction for off-policy data

### 4.6 Multi-Step Training per Rollout

`train()` in `model.py` (line 493) loops over multiple gradient steps per rollout
(controlled by `global_batch_size` relative to `rollout_batch_size * n_samples_per_prompt`):

```python
def train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches):
    for step_id in range(num_steps_per_rollout):
        loss_reduced, grad_norm = train_one_step(
            args, rollout_id, step_id, data_iterator, model, optimizer, opt_param_scheduler,
            num_microbatches[step_id],
        )
```

The number of steps per rollout is:
```
num_steps_per_rollout = (rollout_batch_size * n_samples_per_prompt) / global_batch_size
```

---

## 5. The Sample Type and Data Flow

### 5.1 The `Sample` Dataclass

Defined in `slime/utils/types.py` (line 8), `Sample` is the fundamental data unit that
flows through the entire pipeline:

```python
@dataclass
class Sample:
    # --- Input ---
    group_index: int | None = None        # Which prompt group this belongs to
    index: int | None = None              # Global sample index
    prompt: str | list[dict[str, str]]    # Text prompt or chat message list
    tokens: list[int]                     # Tokenized prompt (and later, prompt+response)
    multimodal_inputs: dict | None        # Raw images/videos for multimodal models

    # --- Generation Output ---
    response: str = ""                    # Generated text
    response_length: int = 0             # Number of generated tokens
    status: Status = PENDING             # COMPLETED, TRUNCATED, ABORTED, FAILED
    rollout_log_probs: list[float]       # Per-token log-probs from rollout engine

    # --- Reward ---
    label: str | None = None             # Ground truth for reward computation
    reward: float | dict | None = None   # Scalar reward or multi-metric dict

    # --- Training Control ---
    loss_mask: list[int] | None = None   # Per-token mask (0 = ignore in loss)
    remove_sample: bool = False          # Flag to exclude from training
    train_metadata: dict | None = None   # Per-sample training overrides (e.g., loss type)

    # --- Metadata ---
    metadata: dict                       # Arbitrary user-defined data
    session_id: str | None = None        # For consistent hashing routing
    weight_versions: list[str]           # Model versions used during generation
```

**Important gotcha:** `Sample.from_dict()` uses `setattr()` to inject arbitrary fields not
in the dataclass schema. Fields like `generate_function_path`, `session_id`, and
`train_metadata` may be set this way -- invisible to type checkers and IDE autocomplete.

### 5.2 Data Flow: Rollout to Training

After rollout generation, `RolloutManager._convert_samples_to_train_data()` (line 683 in
`rollout.py`) converts the list of `Sample` objects into a `RolloutBatch` dict:

```python
train_data = {
    "tokens":              [sample.tokens for sample in samples],
    "response_lengths":    [sample.response_length for sample in samples],
    "total_lengths":       [len(sample.tokens) for sample in samples],
    "loss_masks":          [tensor from sample.loss_mask for sample in samples],
    "rewards":             [normalized_reward for sample in samples],
    "rollout_log_probs":   [tensor from sample.rollout_log_probs for sample in samples],
    "raw_reward":          [original_reward for sample in samples],
    "truncated":           [sample.status == TRUNCATED for sample in samples],
    # ... optional: teacher_log_probs, multimodal_train_inputs, metadata
}
```

This dict is then split across data-parallel ranks by `_split_train_data_by_dp()`.

### 5.3 DataIterator and Micro-Batch Scheduling

`get_data_iterator()` in `slime/backends/megatron_utils/data.py` converts the `RolloutBatch`
dict into `DataIterator` objects that yield micro-batches:

- **Fixed batch size** (`--micro-batch-size`): Each micro-batch has exactly N samples
- **Dynamic batch size** (`--use-dynamic-batch-size`): Samples are packed by token count
  up to `--max-tokens-per-gpu`, allowing variable numbers of samples per micro-batch

The `DataIterator` handles sequence packing (concatenating multiple samples into a single
sequence with `packed_seq_params` for flash attention) and padding.

---

## 6. Adding a New Dataset or Environment

This section walks through adding a new task dataset, using nvidia/Nemotron-Agentic-v1
as an example.

### 6.1 Dataset Format

Slime expects prompt data as a JSONL file (one JSON object per line) or Parquet file.
Each line must contain:

- A **prompt field** (specified by `--input-key`, default `"input"`)
- An optional **label field** (specified by `--label-key`)
- An optional **metadata field** (specified by `--metadata-key`, default `"metadata"`)

**Example: Simple math dataset**
```json
{"input": "What is 2+2?", "label": "4"}
{"input": "Solve: x^2 - 4 = 0", "label": "x = 2 or x = -2"}
```

**Example: Chat-format with `--apply-chat-template`**
```json
{"input": [{"role": "user", "content": "What is 2+2?"}], "label": "4"}
```

**Example: Agentic/tool-calling with tool definitions**
```json
{
  "input": [{"role": "user", "content": "What's the weather in SF?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}],
  "label": "get_weather(location='San Francisco')",
  "metadata": {"task_type": "tool_calling"}
}
```

### 6.2 Step-by-Step: Adding Nemotron-Agentic-v1

#### Step 1: Prepare the Dataset

Download and convert the dataset to JSONL format:

```bash
# Download the dataset
huggingface-cli download nvidia/Nemotron-Agentic-v1 --local-dir /data/nemotron-agentic-v1

# Convert to JSONL (if needed)
python -c "
import json
from datasets import load_dataset
ds = load_dataset('/data/nemotron-agentic-v1')
with open('/data/nemotron-agentic-v1/train.jsonl', 'w') as f:
    for row in ds['train']:
        f.write(json.dumps({
            'input': row['conversations'],  # or the appropriate key
            'label': row['expected_output'],  # ground truth for reward
            'tools': row.get('tools', None),
            'metadata': {'source': 'nemotron-agentic-v1', 'task': row.get('task_type', 'general')},
        }) + '\n')
"
```

Inspect the dataset to identify the correct field names for `--input-key`, `--label-key`,
and `--tool-key`.

#### Step 2: Choose or Implement a Reward Model

For an agentic dataset, you likely need a custom reward function. Create a file, e.g.,
`rewards/nemotron_agentic_rm.py`:

```python
import json

async def custom_rm(args, sample, **kwargs):
    """Reward function for Nemotron-Agentic-v1.
    
    Args:
        args: Training arguments
        sample: Sample with .response (generated text) and .label (ground truth)
    
    Returns:
        float: Reward score
    """
    response = sample.response
    label = sample.label
    
    # Example: Check if the agent produced the correct tool call
    try:
        # Parse tool calls from response
        if label in response:
            return 1.0
        
        # Partial credit for correct function name
        expected_func = json.loads(label)["name"] if isinstance(label, str) else label
        if expected_func in response:
            return 0.5
        
        return 0.0
    except Exception:
        return 0.0
```

The function signature must be:
```python
async def custom_rm(args, sample: Sample, **kwargs) -> float | dict
```

For more complex setups, you can return a dict and use `--reward-key` to select a specific
metric for advantage computation.

#### Step 3: Configure Evaluation Datasets

Create an eval config YAML file, e.g., `configs/nemotron_eval.yaml`:

```yaml
eval:
  defaults:
    max_response_len: 8192
    temperature: 0.7
    top_p: 0.9
  datasets:
    - name: nemotron_agentic_tool_calling
      path: /data/nemotron-agentic-v1/eval_tool_calling.jsonl
      rm_type: custom           # Will use --custom-rm-path
      input_key: input
      label_key: label
      tool_key: tools
      n_samples_per_eval_prompt: 4
      max_response_len: 4096

    - name: nemotron_agentic_planning
      path: /data/nemotron-agentic-v1/eval_planning.jsonl
      rm_type: custom
      input_key: input
      label_key: label
      n_samples_per_eval_prompt: 2
      temperature: 0.5          # Lower temperature for planning tasks
```

The `EvalDatasetConfig` dataclass (in `slime/utils/eval_config.py`) supports all the per-dataset
overrides shown above. Defaults are applied from the `defaults:` section.

#### Step 4: Create the Model Configuration Script

If using Nemotron architecture, create `scripts/models/nemotron-agentic.sh`:

```bash
# Architecture parameters from the HuggingFace config.json
MODEL_ARGS=(
    --swiglu
    --num-layers 40
    --hidden-size 5120
    --ffn-hidden-size 20480
    --num-attention-heads 40
    --group-query-attention
    --num-query-groups 8
    --seq-length 8192
    --max-position-embeddings 32768
    --use-rotary-position-embeddings
    --rotary-base 500000
    --disable-bias-linear
    --normalization "RMSNorm"
    --tokenizer-type HuggingFaceTokenizer
    --bf16
)
```

These values come from the model's `config.json` on HuggingFace. Check `num_hidden_layers`,
`hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, etc.

#### Step 5: Create the Training Launch Script

Create `scripts/run-nemotron-agentic.sh`:

```bash
#!/bin/bash
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/nemotron-agentic.sh"

NUM_GPUS=${NUM_GPUS:-8}

CKPT_ARGS=(
    --hf-checkpoint /data/Nemotron-Agentic-v1              # HF format for SGLang
    --ref-load /data/Nemotron-Agentic-v1_torch_dist        # Megatron format for ref model
    --load /data/nemotron_rl_checkpoints/                   # Resume checkpoint
    --save /data/nemotron_rl_checkpoints/                   # Save checkpoint
    --save-interval 20
)

ROLLOUT_ARGS=(
    --prompt-data /data/nemotron-agentic-v1/train.jsonl    # Training prompts
    --input-key input                                       # JSON key for prompt
    --label-key label                                       # JSON key for ground truth
    --tool-key tools                                        # JSON key for tool definitions
    --apply-chat-template                                   # Use model's chat template
    --rollout-shuffle
    --custom-rm-path rewards/nemotron_agentic_rm.custom_rm  # Custom reward function
    --num-rollout 1000
    --rollout-batch-size 32
    --n-samples-per-prompt 4
    --rollout-max-response-len 4096
    --rollout-temperature 0.8
    --global-batch-size 128
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --eps-clip 0.2
    --eps-clip-high 0.28
)

EVAL_ARGS=(
    --eval-config configs/nemotron_eval.yaml
    --eval-interval 20
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
)

ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats

ray job submit \
    --runtime-env-json='{"env_vars": {"PYTHONPATH": "/path/to/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${NUM_GPUS} \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash
```

#### Step 6: (Optional) Custom Rollout Function

For agentic tasks that require multi-turn interaction (e.g., tool calling with execution
feedback), implement a custom rollout function:

```python
# rollouts/nemotron_agentic_rollout.py

from slime.rollout.base_types import RolloutFnTrainOutput

def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Multi-turn agentic rollout with tool execution."""
    # ... custom logic for multi-turn generation
    # See examples/multi_agent/ and examples/tau-bench/ for reference implementations
    return RolloutFnTrainOutput(samples=all_samples)
```

Wire it via `--rollout-function-path rollouts.nemotron_agentic_rollout.generate_rollout`.

See `examples/multi_agent/` and `examples/tau-bench/` for working multi-turn rollout
implementations.

### 6.3 Checkpoint Conversion

Slime's training uses Megatron-format checkpoints, but SGLang inference uses HuggingFace
format. You need both:

1. **HF checkpoint** (`--hf-checkpoint`): Used by SGLang rollout engines. Download directly
   from HuggingFace.
2. **Megatron `torch_dist` checkpoint** (`--ref-load`): Used by Megatron training actors.
   Converted from HF at first load using the Megatron bridge system
   (`slime_plugins/mbridge/`).

If the model architecture is already supported by an existing bridge (e.g., Llama-based
models use the Llama bridge), no plugin changes are needed. The bridge auto-detects model
type from the HF `config.json`.

---

## 7. Extension Points Reference

Slime provides multiple hook points for customization, all loaded dynamically via
`load_function()` from `slime/utils/misc.py`.

| Argument | Signature | Purpose |
|----------|-----------|---------|
| `--rollout-function-path` | `def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput \| RolloutFnEvalOutput` | Replace the entire rollout pipeline |
| `--custom-rm-path` | `async def custom_rm(args, sample, **kwargs) -> float \| dict` | Custom reward computation |
| `--custom-generate-function-path` | `async def custom_generate(args, sample, sampling_params, evaluation=False) -> Sample` | Per-sample generation override |
| `--custom-reward-post-process-path` | `def post_process(args, samples) -> tuple[list, tensor]` | Custom reward normalization |
| `--dynamic-sampling-filter-path` | `def filter(args, samples, **kwargs) -> DynamicFilterOutput` | Drop groups during rollout |
| `--custom-loss-function-path` | `def custom_loss(args, batch, logits, sum_of_sample_mean) -> tuple[tensor, dict]` | Custom training loss |
| `--data-source-path` | Class with `get_samples(n) -> list[list[Sample]]` | Custom data loading |
| `--buffer-filter-path` | `def filter(buffer, n) -> list[list[Sample]]` | Custom buffer selection |
| `--custom-convert-samples-to-train-data-path` | `def convert(args, samples) -> dict` | Custom sample-to-tensor conversion |
| `--rollout-sample-filter-path` | `def filter(args, sample) -> bool` | Post-generation sample filter |
| `--eval-function-path` | Same as `--rollout-function-path` | Separate function for eval rollouts |
| `--custom-model-provider-path` | `def provider(pre_process, post_process, vp_stage=None) -> GPTModel` | Custom Megatron model |

All paths use Python dotted module notation: `module.submodule.function_name`.
The function is loaded via `importlib` at runtime.

---

## Key Source Files Quick Reference

| Component | File Path |
|-----------|-----------|
| Sync training loop | `train.py` |
| Async training loop | `train_async.py` |
| CLI arguments | `slime/utils/arguments.py` |
| Ray placement groups | `slime/ray/placement_group.py` |
| RolloutManager | `slime/ray/rollout.py` |
| Actor group wrapper | `slime/ray/actor_group.py` |
| Megatron train actor | `slime/backends/megatron_utils/actor.py` |
| Training forward/backward | `slime/backends/megatron_utils/model.py` |
| Loss functions & advantages | `slime/backends/megatron_utils/loss.py` |
| PPO/KL/GAE utilities | `slime/utils/ppo_utils.py` |
| Data iterator | `slime/backends/megatron_utils/data.py` |
| Weight sync (distributed) | `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` |
| SGLang engine actor | `slime/backends/sglang_utils/sglang_engine.py` |
| Default rollout function | `slime/rollout/sglang_rollout.py` |
| Reward model hub | `slime/rollout/rm_hub/__init__.py` |
| Sample dataclass | `slime/utils/types.py` |
| Dataset loading | `slime/utils/data.py` |
| Data source | `slime/rollout/data_source.py` |
| Eval dataset config | `slime/utils/eval_config.py` |
| Rollout base types | `slime/rollout/base_types.py` |
| Dynamic filter hub | `slime/rollout/filter_hub/` |
| Model plugins | `slime_plugins/mbridge/` |
| Example training script | `scripts/run-qwen3-4B.sh` |
| Example eval config | `examples/eval_multi_task/multi_task.yaml` |
