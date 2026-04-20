# 01 · Getting Started

This file shows the minimum steps to start an RL training job in verl: environment setup, the Hydra entrypoint, the anatomy of the default config, and a concrete GSM8K launch script you can run today.

---

## 1. Environment

The repo's `CLAUDE.md` mandates `uv`:

```bash
# Install uv once
curl -LsSf https://astral.sh/uv/install.sh | sh

# Per-project
cd /path/to/verl
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install

# Install verl itself (editable)
uv pip install -e .
# Then install the rollout backend(s) you intend to use:
uv pip install vllm==<pinned>        # see requirements-vllm.txt
# or:  pip install sglang==<pinned>
```

Megatron-LM is *not* pip-installable in a normal way — see `04-engines-megatron-fsdp.md` for how verl pulls it in via Megatron-Core + `AutoBridge`.

---

## 2. The entrypoint

All PPO-family training goes through one file: `verl/trainer/main_ppo.py`.

```python
# verl/trainer/main_ppo.py:36
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)
```

Launch form:

```bash
python -m verl.trainer.main_ppo \
    <group>.<key>=<value> \
    <group>.<key>=<value> ...
```

`run_ppo` initializes Ray, then spawns a `TaskRunner` as a `num_cpus=1` remote actor. The actor runs the whole training script; workers are child Ray actors created later in `RayPPOTrainer.init_workers()`:

```python
# verl/trainer/main_ppo.py:60–100
if not ray.is_initialized():
    ...
    ray.init(**OmegaConf.to_container(ray_init_kwargs))
runner = task_runner_class.remote()
ray.get(runner.run.remote(config))
```

The `TaskRunner.run` method (same file, around `:221`–`:313`) does:

1. Copy model weights to local (faster shm-based loading if `use_shm=True`).
2. Build tokenizer + processor.
3. Build `ResourcePoolManager` for GPU allocation.
4. Build train & val datasets via `create_rl_dataset` (`:316`).
5. Instantiate `RayPPOTrainer` (`:297`), call `init_workers()`, call `fit()`.

---

## 3. Config anatomy — `ppo_trainer.yaml`

The master config file is `verl/trainer/config/ppo_trainer.yaml`. It is a Hydra composition of sub-configs:

```yaml
# (abridged) verl/trainer/config/ppo_trainer.yaml
defaults:
  - model_engine: fsdp           # fsdp | megatron | torchtitan | veomni | mindspeed
  - actor:  <engine-specific>
  - ref:    <engine-specific>
  - critic: <engine-specific>
  - rollout: vllm                # vllm | sglang | trtllm | naive | hf
  - data:   legacy_data
  - reward: ...
  - _self_

algorithm:
  gamma: 1.0
  lam:   1.0
  adv_estimator: gae             # gae | grpo | reinforce_plus_plus | rloo | opo | remax | ...
  use_kl_in_reward: false
  kl_penalty: kl                 # kl | abs | mse | low_var_kl | full | k3 | k3+ | low_var_kl+
  kl_ctrl:
    type: fixed                  # fixed | adaptive
    kl_coef: 0.001

trainer:
  total_epochs: 15
  n_gpus_per_node: 8
  nnodes: 1
  save_freq: 20
  test_freq: 5
  balance_batch: true
```

### Config groups at a glance

| Group | Picks | Key knobs |
|-------|-------|-----------|
| `model_engine` | which `BaseEngine` subclass backs training | `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `param_offload`, `optimizer_offload` |
| `actor` / `critic` / `ref` | per-role engine settings | `optim.lr`, `ppo_mini_batch_size`, `use_dynamic_bsz`, `use_kl_loss` |
| `rollout` | `vllm` / `sglang` / … | `tensor_model_parallel_size`, `gpu_memory_utilization`, `n` (samples per prompt), `free_cache_engine`, `checkpoint_engine.backend`, `checkpoint_engine.update_weights_bucket_megabytes` |
| `data` | dataset/dataloader config | `train_files`, `val_files`, `train_batch_size`, `max_prompt_length`, `max_response_length`, `custom_cls` |
| `reward` | which reward manager + scorer | `reward_manager` (name from `@register`), `reward_fn_key` (default `"data_source"`) |
| `algorithm` | RL math | `adv_estimator`, `kl_penalty`, `use_kl_in_reward`, `kl_ctrl.*` |

### `model_engine` composition

The engine registry in `verl/workers/engine/base.py` (see `EngineRegistry`) maps the string to a concrete class — e.g. `model_engine=megatron` selects `MegatronEngine` at `verl/workers/engine/megatron/transformer_impl.py:71`.

---

## 4. Minimal end-to-end example: GSM8K + Qwen2-7B + SGLang + FSDP

### Step 1 — preprocess the dataset

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

Produces `~/data/gsm8k/train.parquet` and `.../test.parquet`. Each row looks like (`examples/data_preprocess/gsm8k.py:68-84`):

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [{"role": "user", "content": "Janet's ducks ... Let's think step by step and output the final answer after \"####\"."}],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "18"},
  "extra_info": {"split": "train", "index": 0, "answer": "...", "question": "..."}
}
```

These four fields (`data_source`, `prompt`, `reward_model`, `extra_info`) are the canonical schema every dataset must produce — see `06-extending-datasets-and-envs.md`.

### Step 2 — launch

From `examples/ppo_trainer/run_qwen2-7b_sglang_seq_balance.sh`:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="['$HOME/data/gsm8k/train.parquet']" \
    data.val_files="['$HOME/data/gsm8k/test.parquet']" \
    data.train_batch_size=4096 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2-7B-Instruct \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name='qwen2-7b_sglang' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.total_epochs=15
```

### What this does

- `actor_rollout_ref.rollout.name=sglang` + `...tensor_model_parallel_size=2` → SGLang rollout with TP=2.
- `algorithm.adv_estimator=gae` + `critic.*` → use GAE with a critic (classic PPO). Swap to `adv_estimator=grpo` and drop the `critic.*` block to go critic-free.
- `actor_rollout_ref.actor.ppo_mini_batch_size=512` — each PPO update consumes 512 examples at a time; `train_batch_size=4096` → 8 PPO epochs per rollout batch.
- `algorithm.use_kl_in_reward=False` + (by default) `actor.use_kl_loss=...` → KL-vs-ref can be applied as a reward shaper *or* as a loss term; this script does neither. See `05-losses-and-updates.md`.

---

## 5. Where to go for more

- **GRPO (no critic)**: `examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh` (among many).
- **Megatron backend**: `examples/ppo_trainer/run_qwen2-7b_math_gsm8k_megatron.sh`, `examples/grpo_trainer/run_deepseek671b_math_megatron_80gb.sh`.
- **Multi-turn tool use**: `examples/sglang_multiturn/` — tool YAMLs in `examples/sglang_multiturn/config/tool_config/`.
- **Opinionated recipes**: `recipe/dapo/`, `recipe/spin/`, `recipe/gpg/`, etc.

---

## 6. Ray placement (sanity check before launching)

verl's Ray placement is set up by `ResourcePoolManager` at `verl/single_controller/ray/base.py:182`. Default `max_colocate_count=3` packs actor_rollout_ref + rollout + reward-model on the same physical GPUs (colocated mode). You rarely touch this directly — it's implied by the hybrid/standalone/colocated rollout mode (see `03-rollout-and-weight-sync.md`).

If you hit `ray.exceptions.OutOfMemoryError` during launch, double-check `trainer.n_gpus_per_node * trainer.nnodes` ≥ (TP × PP × DP) for *both* training and rollout.
