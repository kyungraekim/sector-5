# 06 · Extending verl: datasets, rewards, tools, agents

verl is deliberately built out of small registries. To plug in a new dataset, reward function, tool, or agent loop, you register a class or branch and point the YAML at it — you do **not** need to edit the trainer.

This file is the cheat-sheet for every extension point: what base class / signature to implement, how to register, and which in-tree file to copy as a template.

---

## 1. Registry cheat-sheet

| Extension | Registration | Anchor |
|---|---|---|
| **Dataset class** | `data.custom_cls.{path,name}` in config → `load_extern_object(path, name)` | `verl/utils/dataset/rl_dataset.py:478` (`get_dataset_class`) |
| **Reward manager** | `@register("name")` on class inheriting `AbstractRewardManager` | `verl/workers/reward_manager/registry.py:24` |
| **Rule-based scorer** (per-dataset) | Add `elif data_source == "...":` branch in `default_compute_score` | `verl/utils/reward_score/__init__.py:44` |
| **Tool** | YAML entry → `initialize_tools_from_config` constructs `class_name(config, tool_schema)` | `verl/tools/utils/tool_registry.py:82` + `verl/tools/base_tool.py:24` |
| **Agent loop** | `@register("agent_name")` on subclass of `AgentLoopBase` | `verl/experimental/agent_loop/agent_loop.py:284,429` |
| **Advantage estimator** | `@register_adv_est(name)` on function | `verl/trainer/ppo/core_algos.py` |
| **Policy-loss variant** | `@register_policy_loss(mode)` on function | `verl/trainer/ppo/core_algos.py` |

Everything below is a per-row deep-dive.

---

## 2. Custom dataset

### 2.1 Canonical row schema

Every verl RL dataset row — whether from a preprocessor or custom class — must expose four fields, shown here from `examples/data_preprocess/gsm8k.py:68-84`:

```python
{
  "data_source": "openai/gsm8k",
  "prompt": [{"role": "user", "content": "<question> Let's think step by step..."}],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "18"},
  "extra_info": {"split": "train", "index": 0, "answer": "...", "question": "..."},
}
```

| Field | Purpose |
|---|---|
| `data_source` | Key that `default_compute_score` dispatches on (see §4). |
| `prompt` | List of chat messages (`role`, `content`, optional `tool_calls`). Gets tokenized via `tokenizer.apply_chat_template`. |
| `reward_model.ground_truth` | Handed to the rule-based scorer. Can be a string, list, or dict. |
| `extra_info` | Free-form dict. Common keys: `tools_kwargs`, `interaction_kwargs`, `tool_selection`, `index`, split label. |

Add images/videos by including them in the message content (verl processes multimodal content through `processor.apply_chat_template`).

### 2.2 When to subclass `RLHFDataset`

The default `RLHFDataset` at `verl/utils/dataset/rl_dataset.py:71` already handles:

- parquet loading + caching,
- chat-template application,
- multimodal extraction (`process_vision_info`),
- per-sample `tools_kwargs` pass-through.

You only need a subclass when you need *non-obvious* hydration — e.g., building a `tools_kwargs` payload on the fly, joining multiple parquet files per row, injecting dataset-wide system prompts.

**Template:**

```python
# my_dataset.py
from verl.utils.dataset.rl_dataset import RLHFDataset

class MyDataset(RLHFDataset):
    def __getitem__(self, idx):
        row = super().__getitem__(idx)
        # Example: derive tools_kwargs per-sample from ground truth.
        row["extra_info"]["tools_kwargs"] = {
            "search": {"create_kwargs": {"corpus_id": row["extra_info"]["corpus_id"]}},
        }
        return row
```

### 2.3 Wiring it in

```yaml
data:
  train_files: ["${oc.env:HOME}/data/mine/train.parquet"]
  val_files:   ["${oc.env:HOME}/data/mine/test.parquet"]
  custom_cls:
    path: /abs/path/to/my_dataset.py
    name: MyDataset
```

The loader is at `get_dataset_class` (`rl_dataset.py:478-505`). It `importlib`-loads the file, fetches the named class, and verifies `issubclass(..., torch.utils.data.Dataset)`. No decorator needed — the YAML itself is the registration.

---

## 3. Reward manager

A **reward manager** is the object that walks the batch, extracts the scalar reward (or token-level reward tensor) per trajectory, and returns it to the trainer. Defaults ship in `verl/workers/reward_manager/`:

| Name | File | Use when |
|---|---|---|
| `naive` | `naive.py` | Most cases — one scalar per trajectory, placed on EOS token. |
| `batch` | `batch.py` | Vectorized scoring (score the whole batch at once, e.g., remote API). |
| `prime` | `prime.py` | Structured PRM-style scoring. |
| `dapo` | `dapo.py` | DAPO-specific reward shaping. |

### 3.1 Contract

Implement `AbstractRewardManager.__call__(data: DataProto) -> dict` (`verl/workers/reward_manager/abstract.py`). The minimum return shape is:

```python
{
    "reward_tensor": torch.FloatTensor,   # (B, T) — usually zero except on EOS
    "reward_extra_info": {"key": list[Any], ...},  # logged per-sample
}
```

### 3.2 Register and select

```python
# verl/workers/reward_manager/my_manager.py
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import register

@register("my_manager")
class MyRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score  # falls back to default_compute_score

    def __call__(self, data):
        ...
```

Select via `reward_model.reward_manager=my_manager`. The registry (`verl/workers/reward_manager/registry.py:24`) raises on duplicate names — importing the file once at start-up is enough for `@register` to bind.

### 3.3 When to write a new manager vs. a new scorer

- **New scorer** (§4): per-dataset final-answer grading. 99% of the time this is all you need.
- **New manager**: per-step shaped rewards, asynchronous remote grading, multi-trajectory aggregation (e.g., tournament style), token-level reward tensors.

---

## 4. Rule-based scorer (per-dataset)

The entry point is `default_compute_score(data_source, solution_str, ground_truth, extra_info, ...)` at `verl/utils/reward_score/__init__.py:19`. It dispatches by `data_source`:

```python
# verl/utils/reward_score/__init__.py:44-107
if data_source == "openai/gsm8k":
    from . import gsm8k
    res = gsm8k.compute_score(solution_str, ground_truth)
elif data_source in ["lighteval/MATH", ...]:
    from . import math_reward
    res = math_reward.compute_score(...)
...
else:
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
```

### 4.1 Adding a scorer

1. Create `verl/utils/reward_score/my_task.py`:

    ```python
    def compute_score(solution_str: str, ground_truth) -> float | dict:
        # return float in [0, 1] OR a dict like {"score": 0.7, "acc": 1.0, "format": 0.5}
        ...
    ```

2. Branch in `default_compute_score`:

    ```python
    elif data_source == "nvidia/Nemotron-Agentic-v1":
        from . import nemotron
        res = nemotron.compute_score(solution_str, ground_truth, extra_info=extra_info)
    ```

The return value may be:
- `float` / `int` / `bool` — cast to float.
- `dict` — returned verbatim; extra keys show up in `reward_extra_info` logs.
- `tuple` — first element used as the score.

### 4.2 Bypassing the default

If you don't want to touch `verl/utils/reward_score/__init__.py`, write a `custom_reward_function.py` and point Hydra at it:

```yaml
custom_reward_function:
  path: /abs/path/my_reward.py
  name: compute_score
```

The reward manager prefers this over `default_compute_score` when present.

---

## 5. Tools

Tools are what the model calls during multi-turn agentic rollout (search, code exec, MCP, browser, …). Two contracts matter: the class, and the YAML that instantiates it.

### 5.1 The `BaseTool` contract

From `verl/tools/base_tool.py:24-93`:

```python
class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema): ...

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema: ...

    async def create(self, instance_id: Optional[str] = None, **kwargs
                     ) -> tuple[str, ToolResponse]:
        """Per-trajectory setup. Return the instance id + optional opening message."""

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs
                      ) -> tuple[ToolResponse, float, dict]:
        """Returns (response, step_reward, metrics). `parameters` comes from the model's JSON."""

    async def calc_reward(self, instance_id: str, **kwargs) -> float: ...
    async def release(self, instance_id: str, **kwargs) -> None: ...
```

`ToolResponse` (see `verl/tools/schemas.py`) carries `text`, `image`, or `video`. The step reward is summed into `reward_tensor` if your reward manager is step-aware.

### 5.2 In-tree templates

| Pattern | File |
|---|---|
| Minimal rule-graded tool | `verl/tools/gsm8k_tool.py` |
| HTTP search tool | `verl/tools/search_tool.py` |
| Sandboxed code execution | `verl/tools/sandbox_fusion_tools.py` |
| Generic MCP client | `verl/tools/mcp_base_tool.py` |
| Browser / web | `verl/tools/browser_tool.py` |

### 5.3 YAML wiring

From `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`:

```yaml
tools:
  - class_name: verl.tools.gsm8k_tool.Gsm8kTool
    config:
      type: native
    tool_schema:
      type: function
      function:
        name: calc_gsm8k_reward
        description: "Submit your best answer to check correctness."
        parameters:
          type: object
          properties:
            answer: {type: string, description: "Final numeric answer."}
          required: [answer]
```

`initialize_tools_from_config` at `verl/tools/utils/tool_registry.py:82` does:

1. Parse `tools_config_file` (YAML).
2. For each entry: resolve `class_name` via `get_tool_class` (dynamic `importlib`).
3. Build `OpenAIFunctionToolSchema` (or fetch it via MCP for `type: mcp`).
4. Instantiate `tool_cls(config=..., tool_schema=...)`.

Point the trainer at the YAML:

```yaml
actor_rollout_ref.rollout.multi_turn.tool_config_path: /abs/path/tools.yaml
actor_rollout_ref.rollout.multi_turn.format: hermes   # or qwen3, llama3_json, etc.
```

### 5.4 Per-sample `tools_kwargs`

A dataset row can pre-seed per-trajectory tool state via `extra_info.tools_kwargs`:

```json
{
  "extra_info": {
    "tools_kwargs": {
      "calc_gsm8k_reward": {
        "create_kwargs": {"ground_truth": "18"},
        "execute_kwargs": {}
      }
    }
  }
}
```

`create_kwargs` is forwarded to `BaseTool.create(...)`, `execute_kwargs` to every `execute(...)` call. Use this for ground-truth that only the grader may see.

---

## 6. Agent loop

An **agent loop** is the state machine that drives one rollout trajectory: observe → think → maybe-call-tool → observe → … . Default loops:

| Name | File | Behavior |
|---|---|---|
| `single_turn_agent` | `single_turn_agent_loop.py` | One generation, no tools. Baseline for non-agentic tasks. |
| `tool_agent` | `tool_agent_loop.py` | Tool-calling loop: generate → parse tool calls → execute → repeat until EOS or `max_assistant_turns`. |
| `diffusion_single_turn_agent` | `diffusion_single_turn_agent_loop.py` | Single-turn for diffusion-LM backbones. |

### 6.1 Contract

From `verl/experimental/agent_loop/agent_loop.py:284`:

```python
class AgentLoopBase(ABC):
    def __init__(self, trainer_config, server_manager, tokenizer,
                 processor, dataset_cls, data_config, **kwargs): ...

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """kwargs contains per-row dataset fields (raw_prompt, tools_kwargs, ...)."""
```

`AgentLoopOutput` carries `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, `multi_modal_data`, `num_turns`, `metrics`, …  — the fields the trainer needs to build the PPO batch.

### 6.2 Register and select

```python
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register

@register("my_agent")
class MyAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        ...
```

`register` (`agent_loop.py:429`) records `{"_target_": "module.MyAgentLoop"}` in `_agent_loop_registry`, and Hydra `instantiate`s it when the server manager selects the agent.

Select in config:

```yaml
actor_rollout_ref.rollout.multi_turn.agent.agent_name: my_agent
```

### 6.3 When to subclass

- Non-standard turn topology (multi-agent conversation, tree-of-thought, planner-executor).
- Custom tool-call parsing beyond what `ToolParser` supports.
- Reward shaping that reads intra-rollout state (partial scores between turns).

For anything that is "tool-calling, but with different tools," just reuse `tool_agent` and swap the tool YAML — don't write a new loop.

---

## 7. Advantage estimator

Advantage estimators live in `verl/trainer/ppo/core_algos.py` and are selected by `algorithm.adv_estimator`. Built-ins: `gae`, `grpo`, `reinforce_plus_plus`, `rloo`, `opo`, `remax`, `reinforce_plus_plus_baseline`, `grpo_passk`, …

### 7.1 Register

```python
from verl.trainer.ppo.core_algos import register_adv_est, AdvantageEstimator

@register_adv_est("my_adv")
def compute_my_advantage(token_level_rewards, response_mask, config, **kwargs):
    # returns advantages, returns  — both (B, T) tensors
    ...
```

Then `algorithm.adv_estimator=my_adv` and (for non-GAE estimators) drop the critic block.

---

## 8. Policy-loss variant

Loss modes sit behind `actor.policy_loss.loss_mode` (`vanilla`, `grpo`, `dapo`, `reinforce_plus_plus`, `opo`, …). Built-ins cover most of the literature; for a new variant:

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_loss")
def my_policy_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config):
    # returns pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
    ...
```

See `05-losses-and-updates.md` §4.2 for the full signature and an in-tree example. Select with `actor.policy_loss.loss_mode=my_loss`.

---

## 9. Checklist for a new task end-to-end

1. **Data schema** — preprocess to the four canonical fields; decide what goes in `extra_info`.
2. **Scorer** — add a branch in `default_compute_score` (or write `custom_reward_function`).
3. **Reward manager** — default `naive` unless you need step-level rewards.
4. **Tools** — write a `BaseTool` subclass per tool; wire up YAML.
5. **Agent loop** — reuse `tool_agent` unless the turn structure is genuinely new.
6. **Config** — override `ppo_trainer.yaml` with the custom dataset class, tool config path, reward manager name, advantage estimator.
7. **Launch** — `python -m verl.trainer.main_ppo --config-name=<your_config> ...`.

`07-nemotron-agentic-v1-walkthrough.md` walks this list end-to-end for a concrete dataset.
