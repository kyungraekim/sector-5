# 07 · End-to-end walkthrough: `nvidia/Nemotron-Agentic-v1`

This file takes everything from `01`–`06` and stitches it into a runnable recipe for a new agentic dataset: **`nvidia/Nemotron-Agentic-v1`**. By the end you should have:

1. a preprocessor that emits verl-shaped parquet,
2. a dataset class (only if needed),
3. one or more tools implementing `BaseTool`,
4. a rule-based scorer and a reward-manager selection,
5. a config that inherits `ppo_trainer.yaml`,
6. a launch command, and
7. a smoke-test checklist.

> **TODO — confirm the HuggingFace schema.** This guide was written without access to HuggingFace, so the Nemotron row shape is **assumed**: each example has `messages: list[{role, content, tool_calls?}]` plus a verifiable answer or trajectory label. If the real dataset exposes different field names (for example `conversation`, `turns`, `expected_answer`), adjust §2 and §3 accordingly. Everything downstream is schema-agnostic once the four canonical verl fields are populated.

---

## 1. Assumed schema (what the guide pretends the dataset looks like)

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful agent. You may call tools."},
    {"role": "user",      "content": "What was GDP of France in 2022?"},
    {"role": "assistant", "content": "", "tool_calls": [{"name":"web_search","arguments":{"q":"France GDP 2022"}}]},
    {"role": "tool",      "content": "World Bank: 2.78 trillion USD", "name": "web_search"},
    {"role": "assistant", "content": "2.78T USD (World Bank, 2022)."}
  ],
  "answer": "2.78 trillion USD",
  "tools_available": ["web_search", "calculator"]
}
```

Mapping to verl's canonical four fields (§2 of `06-extending-datasets-and-envs.md`):

| verl field | Source in the Nemotron row |
|---|---|
| `data_source` | hard-coded `"nvidia/Nemotron-Agentic-v1"` |
| `prompt` | `messages` up to the *first assistant turn* (system + user) |
| `reward_model.ground_truth` | `answer` (final verifiable answer) |
| `extra_info.ref_trajectory` | `messages` from the first assistant turn onward — keep for offline eval |
| `extra_info.tools_kwargs` | Derived per-sample; see §3 |
| `extra_info.tool_selection` | `tools_available` — restricts the global tool list per-sample |

---

## 2. Preprocess the dataset

Create `examples/data_preprocess/nemotron_agentic_v1.py`. Pattern cribbed from `examples/data_preprocess/gsm8k_multiturn_w_tool.py:60-106`.

```python
# examples/data_preprocess/nemotron_agentic_v1.py
import argparse, os
import datasets
from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "nvidia/Nemotron-Agentic-v1"

def split_prompt_and_trajectory(messages):
    """Seed = system+user before the first assistant turn. Rest = ref trajectory."""
    first_assistant = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
    return messages[:first_assistant], messages[first_assistant:]

def make_row(example, idx, split):
    messages = example["messages"]           # TODO: confirm field name
    answer   = example["answer"]             # TODO: confirm field name
    tools    = example.get("tools_available", [])

    prompt, ref_traj = split_prompt_and_trajectory(messages)

    # tools_kwargs is consumed by BaseTool.create(); use it to smuggle ground truth.
    tools_kwargs = {
        "final_answer_check": {
            "create_kwargs": {"ground_truth": answer},
        }
    }

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "agentic",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
            "tool_selection": tools,        # ToolAgentLoop honors this (§5.4 of 06-*)
            "ref_trajectory": ref_traj,     # not fed to the model; kept for eval
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", required=True)
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    ds = datasets.load_dataset(DATA_SOURCE)   # splits: assume "train" / "validation"
    train = ds["train"].map(lambda ex, idx: make_row(ex, idx, "train"),
                            with_indices=True, remove_columns=ds["train"].column_names)
    val   = ds["validation"].map(lambda ex, idx: make_row(ex, idx, "val"),
                                 with_indices=True, remove_columns=ds["validation"].column_names)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    val.to_parquet(os.path.join(args.local_save_dir, "val.parquet"))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_save_dir, dst=args.hdfs_dir)

if __name__ == "__main__":
    main()
```

Run it:

```bash
python examples/data_preprocess/nemotron_agentic_v1.py \
  --local_save_dir $HOME/data/nemotron_agentic_v1
```

---

## 3. Custom dataset class — usually skip

`RLHFDataset` already handles everything above. Write a subclass only if you need per-item logic (e.g., filling `tools_kwargs` from a column you kept around). Template:

```python
# nemotron_dataset.py
from verl.utils.dataset.rl_dataset import RLHFDataset

class NemotronAgenticDataset(RLHFDataset):
    def __getitem__(self, idx):
        row = super().__getitem__(idx)
        # Example: if you kept `corpus_id` alongside messages, hydrate search tool kwargs.
        corpus_id = row["extra_info"].get("corpus_id")
        if corpus_id is not None:
            row["extra_info"].setdefault("tools_kwargs", {}).setdefault(
                "web_search", {"create_kwargs": {"corpus_id": corpus_id}}
            )
        return row
```

Wiring:

```yaml
data:
  custom_cls:
    path: /abs/path/nemotron_dataset.py
    name: NemotronAgenticDataset
```

For the default schema above, this subclass isn't required — skip it.

---

## 4. Tools

Assume Nemotron needs two tools: `web_search` and `calculator`. Both follow the `BaseTool` contract (`verl/tools/base_tool.py:24`).

### 4.1 `web_search` — can reuse the in-tree template

`verl/tools/search_tool.py` already implements an HTTP-backed search. If your backend speaks the same JSON API, just register it in YAML (no code). Otherwise, subclass:

```python
# verl/tools/nemotron_web_search.py
from verl.tools.base_tool import BaseTool, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

class NemotronWebSearch(BaseTool):
    async def create(self, instance_id=None, **kwargs):
        instance_id, _ = await super().create(instance_id)
        # kwargs comes from extra_info.tools_kwargs[<tool_name>].create_kwargs
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id, parameters, **kwargs):
        query = parameters.get("q", "")
        results = await self._call_backend(query)      # your HTTP client
        return ToolResponse(text=results), 0.0, {"num_hits": len(results)}

    async def _call_backend(self, query): ...
```

### 4.2 `calculator` — minimal example

```python
# verl/tools/nemotron_calculator.py
import asteval
from verl.tools.base_tool import BaseTool, ToolResponse

class NemotronCalculator(BaseTool):
    async def execute(self, instance_id, parameters, **kwargs):
        try:
            value = asteval.Interpreter()(parameters["expression"])
            return ToolResponse(text=str(value)), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"error: {e}"), 0.0, {"error": 1}
```

### 4.3 Tool YAML

Create `configs/tools/nemotron_tools.yaml` (template: `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`):

```yaml
tools:
  - class_name: verl.tools.nemotron_web_search.NemotronWebSearch
    config:
      type: native
      endpoint: http://search.internal/v1/query   # example
    tool_schema:
      type: function
      function:
        name: web_search
        description: "Query the web."
        parameters:
          type: object
          properties:
            q: {type: string, description: "Search query."}
          required: [q]

  - class_name: verl.tools.nemotron_calculator.NemotronCalculator
    config: {type: native}
    tool_schema:
      type: function
      function:
        name: calculator
        description: "Evaluate an arithmetic expression."
        parameters:
          type: object
          properties:
            expression: {type: string}
          required: [expression]
```

`initialize_tools_from_config` (`verl/tools/utils/tool_registry.py:82`) will import each `class_name` and build a `tool_schema`. At runtime, per-sample `extra_info.tool_selection` restricts which of these fire.

---

## 5. Reward

### 5.1 Scorer (`verl/utils/reward_score/nemotron.py`)

```python
# verl/utils/reward_score/nemotron.py
import re

def _extract_final(s: str) -> str:
    # Adapt to the actual answer format. Assume final answer after "####".
    m = re.search(r"####\s*(.+)$", s.strip())
    return m.group(1).strip() if m else s.strip()

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def compute_score(solution_str: str, ground_truth, extra_info=None) -> dict:
    pred = _normalize(_extract_final(solution_str))
    gt   = _normalize(ground_truth if isinstance(ground_truth, str) else str(ground_truth))
    exact = float(pred == gt)
    return {
        "score": exact,
        "acc": exact,
        "pred": pred,
    }
```

Branch into `default_compute_score` at `verl/utils/reward_score/__init__.py:44`:

```python
elif data_source == "nvidia/Nemotron-Agentic-v1":
    from . import nemotron
    res = nemotron.compute_score(solution_str, ground_truth, extra_info=extra_info)
```

### 5.2 Reward manager

`naive` is fine unless you want per-step tool rewards rolled in. Select via:

```yaml
reward_model:
  reward_manager: naive
  reward_fn_key: data_source
```

---

## 6. Agent loop

Reuse the built-in `tool_agent` (`verl/experimental/agent_loop/tool_agent_loop.py:88`) — it honors per-sample `tool_selection`, handles parallel tool calls, and respects `max_assistant_turns`.

Only write a custom `@register("nemotron_agent")` loop if Nemotron requires a turn topology `tool_agent` doesn't cover (e.g., a planner-executor hand-off or mandatory reflection turns). See `06-*.md §6` for the base class.

---

## 7. Config — overrides to `ppo_trainer.yaml`

Save as `verl/trainer/config/nemotron_agentic_v1.yaml`. It inherits `ppo_trainer.yaml` and overrides the pieces that differ:

```yaml
# @package _global_
defaults:
  - ppo_trainer
  - _self_

data:
  train_files:
    - ${oc.env:HOME}/data/nemotron_agentic_v1/train.parquet
  val_files:
    - ${oc.env:HOME}/data/nemotron_agentic_v1/val.parquet
  train_batch_size: 512
  max_prompt_length: 4096
  max_response_length: 8192
  return_raw_chat: true       # required for multi-turn / tool calling
  # custom_cls:               # uncomment only if you wrote one in §3
  #   path: /abs/path/nemotron_dataset.py
  #   name: NemotronAgenticDataset

actor_rollout_ref:
  model:
    path: Qwen/Qwen3-30B-A3B    # example MoE — swap as needed
    use_remove_padding: true
  actor:
    ppo_mini_batch_size: 128
    use_dynamic_bsz: true
    use_kl_loss: true
    kl_loss_coef: 0.001
    optim: {lr: 1e-6}
  rollout:
    name: sglang                # or vllm
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.55
    n: 4                        # samples per prompt — GRPO-friendly
    multi_turn:
      tool_config_path: /abs/path/configs/tools/nemotron_tools.yaml
      format: hermes            # match the tokenizer's tool-call template
      max_assistant_turns: 8
      max_user_turns: 8
      max_parallel_calls: 2
      agent:
        agent_name: tool_agent  # or nemotron_agent if you wrote a custom loop

reward_model:
  reward_manager: naive
  reward_fn_key: data_source

algorithm:
  adv_estimator: grpo           # critic-free — drop critic.* section
  use_kl_in_reward: false       # KL is applied in the loss (use_kl_loss=true above)
  kl_penalty: low_var_kl

trainer:
  total_epochs: 2
  n_gpus_per_node: 8
  nnodes: 1
  save_freq: 50
  test_freq: 10
  balance_batch: true
  project_name: verl_nemotron_agentic_v1
  experiment_name: qwen3_30b_moe_grpo
  logger: ["console", "wandb"]
```

### Engine pick

- **Megatron** (`model_engine=megatron`) is the primary target for >30B MoE (see `04-engines-megatron-fsdp.md` for TP/PP/EP knobs). Add:

  ```yaml
  model_engine: megatron
  actor_rollout_ref.actor.megatron:
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 2
    expert_model_parallel_size: 4
  ```

- **FSDP** (`model_engine=fsdp`, the default) is simpler for ≤14B dense models. No extra config needed.

---

## 8. Launch

```bash
python -m verl.trainer.main_ppo \
  --config-name=nemotron_agentic_v1 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=2
```

Smoke-test first (single node, small sample):

```bash
python -m verl.trainer.main_ppo \
  --config-name=nemotron_agentic_v1 \
  data.train_files='["'"$HOME"'/data/nemotron_agentic_v1/train.parquet"]' \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=2 \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1
```

---

## 9. Verification checklist

Before scaling up, verify the plumbing end-to-end:

1. **Data** — open the parquet in Python, assert the four canonical fields on a random row, assert `extra_info.tools_kwargs` is non-empty where expected.
2. **Tool instantiation** — start a short run with `logger=console`, confirm the tool-YAML print at startup lists both `web_search` and `calculator`.
3. **Tool actually fires** — enable rollout tracing (`docs/advance/rollout_trace.rst`) and grep the trace for `tool_call_id`. If zero calls happen, the model's chat template doesn't match `multi_turn.format` — flip `format` to `qwen3`, `llama3_json`, etc., until the parser recognizes the tool calls.
4. **Weight-sync cadence** — rollout trace should show, per step, the sequence
   `rollout.release(weights) → get_per_tensor_param → rollout.update_weights → rollout.resume(kv_cache)` in that order (cross-reference `03-rollout-and-weight-sync.md §4`).
5. **Scorer wiring** — check that `reward/score` in W&B is non-trivially non-zero within the first few steps. Zero across every sample usually means `data_source` doesn't match the branch in `default_compute_score`.
6. **KL health** — `actor/kl_loss` should stay small (single-digit × `kl_loss_coef`). Exploding KL → lower LR, raise `kl_loss_coef`, or shorten `max_response_length`.

---

## 10. Things to double-check against the real HF dataset

Before shipping this to your team, verify with the actual `nvidia/Nemotron-Agentic-v1`:

- **Field names** — `messages`? `conversation`? `turns`? Is the answer `answer`, `expected_output`, or embedded in a final message?
- **Tool schemas** — does the dataset ship its own tool definitions, or is the tool set implicit?
- **Ground truth granularity** — final-answer only, full-trajectory label, or per-turn step rewards?
- **System prompt** — is there a canonical system prompt you must preserve (vs. the one you build)?
- **Splits** — `train`/`validation`/`test`? Any leakage between prompt seeds?

Update §1–§5 once confirmed. The rest of the pipeline (config, launch, verification) doesn't change.

---

## 11. Where to go next

- **Scaling to multiple nodes** — `04-engines-megatron-fsdp.md` for TP/PP/EP sizing rules.
- **Debugging stalled rollouts** — `03-rollout-and-weight-sync.md §9` (weight-sync debugging checklist).
- **Writing custom losses** — `05-losses-and-updates.md §4.2`.
- **Adding MCP tools instead of native** — `verl/tools/utils/tool_registry.py:36` (`initialize_mcp_tool`) and `verl/tools/mcp_base_tool.py`.
