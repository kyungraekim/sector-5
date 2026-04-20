# verl — Architecture & Extension Guide

A code-level tour of [verl-project/verl](https://github.com/verl-project/verl), written for engineers and ML researchers who need to operate, debug, or extend the framework rather than just read the top-level README.

Paths in this guide are relative to the verl repo root (`/Users/kyungrae/workspace/dev/claude-code/verl` on the author's machine). Line numbers are from `main` @ commit `63dc0850` (2026-04-21). They may drift slightly as the codebase evolves — trust the symbol name over the line number if they disagree.

---

## One-paragraph summary

**verl is a Ray-orchestrated RLHF / agentic-RL framework that structurally separates a *training engine* (Megatron-LM, FSDP, torchtitan, veomni, mindspeed) from an *inference/rollout engine* (SGLang, vLLM, TRT-LLM, HF, naive).** Per training step, the trainer puts the rollout engine to *sleep* (frees KV cache + weights on the rollout GPUs), runs the policy update via the training engine, then *wakes* the rollout and ships updated weights over **CUDA IPC** in bucketed chunks. The same pattern supports colocated (same GPU), hybrid (same node, separate processes), and standalone (separate clusters over NCCL/NIXL) deployments. On top of this, an async `AgentLoop` layer supports multi-turn tool-using agents with a sticky-session load balancer for prefix-cache reuse.

---

## Read in this order

| # | File | Who cares |
|---|------|-----------|
| 01 | [`01-getting-started.md`](./01-getting-started.md) | Anyone launching a job |
| 02 | [`02-training-loop.md`](./02-training-loop.md) | ML researchers debugging reward / advantage / update |
| 03 | [`03-rollout-and-weight-sync.md`](./03-rollout-and-weight-sync.md) | Infra engineers, anyone hitting OOM or weight-sync stalls |
| 04 | [`04-engines-megatron-fsdp.md`](./04-engines-megatron-fsdp.md) | Anyone tuning parallelism (TP/PP/EP/CP) |
| 05 | [`05-losses-and-updates.md`](./05-losses-and-updates.md) | ML researchers writing new losses / advantage estimators |
| 06 | [`06-extending-datasets-and-envs.md`](./06-extending-datasets-and-envs.md) | Anyone adding a dataset / reward / tool / agent loop |
| 07 | [`07-nemotron-agentic-v1-walkthrough.md`](./07-nemotron-agentic-v1-walkthrough.md) | Concrete end-to-end worked example |

If you want the single shortest path to "I understand how one training step works": skim `02`, then read `03` end-to-end — `03` is where the actual RL infrastructure lives.

---

## Top-level repo map

Only directories you'll touch regularly:

```
verl/
├── trainer/
│   ├── main_ppo.py              # Hydra entrypoint; `run_ppo`, `TaskRunner`
│   ├── config/                  # All default YAMLs
│   │   └── ppo_trainer.yaml     # Master config
│   ├── ppo/
│   │   ├── ray_trainer.py       # RayPPOTrainer.fit() — the training loop
│   │   ├── core_algos.py        # apply_kl_penalty, compute_advantage, policy-loss registry
│   │   └── reward.py
│   └── constants_ppo.py
├── workers/
│   ├── engine_workers.py        # ActorRolloutRefWorker — owns train engine + rollout, syncs weights
│   ├── engine/
│   │   ├── base.py              # BaseEngine contract
│   │   ├── megatron/            # Megatron-LM training backend (primary)
│   │   ├── fsdp/                # FSDP training backend (most examples)
│   │   ├── torchtitan/ veomni/ mindspeed/
│   │   └── automodel/
│   ├── rollout/
│   │   ├── sglang_rollout/      # SGLang adapter
│   │   ├── vllm_rollout/        # vLLM adapter + bucketed IPC sender/receiver
│   │   ├── trtllm_rollout/ naive/ hf_rollout.py
│   │   └── replica.py           # RolloutMode enum (COLOCATED / HYBRID / STANDALONE)
│   ├── reward_manager/          # registry.py — @register("naive"), @register("dapo"), ...
│   └── utils/losses.py          # ppo_loss, sft_loss, value_loss
├── experimental/
│   ├── agent_loop/              # Async multi-turn agent + tool runner
│   │   ├── agent_loop.py        # AgentLoopBase, AsyncLLMServerManager, LoadBalancer, @register
│   │   ├── tool_agent_loop.py   # @register("tool_agent")
│   │   └── single_turn_agent_loop.py
│   └── reward_loop/             # Parallel reward-model inference path
├── tools/
│   ├── base_tool.py             # BaseTool contract
│   ├── gsm8k_tool.py search_tool.py sandbox_fusion_tools.py geo3k_tool.py image_zoom_in_tool.py
│   └── utils/tool_registry.py   # initialize_tools_from_config
├── utils/
│   ├── dataset/rl_dataset.py    # RLHFDataset + get_dataset_class (custom_cls hook)
│   └── reward_score/__init__.py # default_compute_score — rule-based scorers routed by data_source
├── checkpoint_engine/
│   └── base.py                  # CheckpointEngineManager (the thing fit() actually calls)
├── single_controller/
│   └── ray/base.py              # RayWorkerGroup, ResourcePoolManager
└── models/mcore/                # Megatron-Core model wrappers / bridges

recipe/                          # Opinionated launch scripts (DAPO, SPIN, GRPO recipes, ...)
examples/
├── data_preprocess/             # HF → parquet converters
├── ppo_trainer/ grpo_trainer/ sglang_multiturn/ ...
└── sglang_multiturn/config/tool_config/   # Tool YAML templates
```

---

## Two critical call paths to anchor on

Memorizing these two paths makes the rest of the codebase navigable:

**1. Training step (driver side).**
```
verl/trainer/main_ppo.py:36   @hydra.main → run_ppo
    → TaskRunner.run
        → RayPPOTrainer(...).fit()                                 # verl/trainer/ppo/ray_trainer.py:1260
            → async_rollout_manager.generate_sequences(...)        #                           :1351
            → self._compute_old_log_prob(batch)                    #                           :1434
            → self._compute_ref_log_prob(batch)                    #                           :1470
            → self._compute_values(batch)                          #                           :1476
            → apply_kl_penalty(batch, ...)                         # trainer/ppo/core_algos.py:76
            → compute_advantage(batch, adv_estimator, ...)         #                          :136
            → self._update_critic(batch) / self._update_actor(batch)
            → self.checkpoint_manager.update_weights(global_steps) # ray_trainer.py:1566
```

**2. Weight sync (worker side).**
```
CheckpointEngineManager.update_weights           # verl/checkpoint_engine/base.py:409
  ├─ backend == "naive" (sync/colocated)
  │    → trainer.update_weights() (Ray RPC)
  │        → ActorRolloutRefWorker.update_weights # verl/workers/engine_workers.py:663
  │            1. rollout.resume(tags=["weights"])
  │            2. actor.engine.get_per_tensor_param(...)          # generator of (name, tensor)
  │            3. rollout.update_weights(generator, ...)          # vllm_rollout.py:162 / sglang_rollout.py:205
  │                 └→ BucketedWeightSender → ZMQ → update_weights_from_ipc (receiver) → CUDA IPC / shm
  │            4. actor.engine.to("cpu", ...)                     # offload trainer
  │            5. rollout.resume(tags=["kv_cache"])
  └─ backend in ("nccl", "nixl", ...) (async/disaggregated)
       → trainer-side checkpoint_engine.send_weights(per_tensor_param)
       → rollout-side CheckpointEngineWorker.update_weights → server_adapter.update_weights
```

---

## A note on the "engine refactor"

The recent commits `044bbba2 [BREAKING] [misc] refactor: deprecate workers, migrate to engines` and `63dc0850 [BREAKING] [env] refactor: deprecate verl/interactions` consolidated what used to be multiple role-specific workers (actor worker, critic worker, ref worker) into a single `ActorRolloutRefWorker` (`verl/workers/engine_workers.py`) that internally composes:

- `self.actor` — a `TrainingWorker` wrapping a `BaseEngine` (Megatron/FSDP/...)
- `self.ref` — another `TrainingWorker` for the reference policy
- `self.rollout` — a `BaseRollout` (SGLang/vLLM/...)

If you're reading older tutorials or blog posts that reference `MegatronActor`, `FSDPCritic`, etc., those roles still exist conceptually but are now selected via the `model_engine: {fsdp, megatron, torchtitan, veomni, mindspeed}` config group instead of by worker class.

Similarly, `verl/interactions` (deprecated in `63dc0850`) is superseded by `verl/experimental/agent_loop/` + `verl/tools/`. Any doc referencing `Interaction` classes is stale.
