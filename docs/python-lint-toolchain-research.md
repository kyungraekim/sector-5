# Python Linting & Formatting Toolchain: PyTorch Deep Dive + Cross-Project Comparison

## PyTorch's Setup

### Tools

| Category | Tool | Version | Notes |
|----------|------|---------|-------|
| Linter | **Ruff** | via lintrunner | Primary Python linter, config in `pyproject.toml` |
| Linter | **Flake8** | via lintrunner | Legacy, still in CI alongside Ruff |
| Formatter | **Ruff format** | (same) | Double quotes, docstring code formatting |
| Import sorting | **isort** + **usort** | — | isort uses `profile = "black"`, usort in PYFMT linter |
| Type checker | **Pyrefly** | 0.52.0 | Modern Meta replacement, used in CI |
| Type checker | **Mypy** | — | Legacy, strict mode (`disallow_untyped_defs = True`, Python 3.11) |
| C/C++ | **clang-format**, **clang-tidy** | — | For ATen/c10 code |
| Spelling | **codespell** | — | Catches typos in comments/strings |
| Orchestrator | **lintrunner** | 0.12.7 | Runs all linters uniformly (~55 linter definitions) |
| CLI wrapper | **spin** | — | `spin lint`, `spin fixlint`, `spin quicklint` commands |

### Configuration Files

- **`pyproject.toml`** — Ruff rules (line-length=88, double quotes, docstring-code-format), isort config (profile="black"), usort config
- **`.flake8`** — Flake8 config (max-line-length=120, selects B/C/E/F/G/P/SIM/T4/W/B9)
- **`.lintrunner.toml`** — Defines ~55 linter commands with file patterns and exclusions
- **`mypy.ini`** — Python 3.11, `disallow_untyped_defs=True`, per-module overrides (inductor/dynamo get `disallow_any_generics=True`; caffe2/quantization/testing get `ignore_errors=True`)
- **`pyrefly.toml`** — Python 3.12, sub-configs for `_dynamo/`, `_dispatch/`, `_subclasses/`, `_functorch/`
- **`.spin/cmds.py`** — Defines lint/fixlint/quicklint commands

### Key Design Decisions

1. **Line length: dual threshold** — Ruff format wraps at **88** (soft target), but E501 is explicitly ignored in both Ruff and Flake8. The actual hard lint limit is **Flake8 B950 at 120** (listed in Ruff's `external` list, enforced only by Flake8). B950 is more lenient than E501 — it allows ~10% overshoot, so lines up to ~132 may pass. Lines between 88-120 are common and accepted.
2. **Qualified annotations required** — `# type: ignore[code]` and `# noqa: CODE` must specify error codes (enforced by TYPEIGNORE and NOQA linters)
3. **Performance-tiered execution** — linters classified as Very Fast, Fast, Slow; fast ones run on all files, slow ones (Flake8, clang-format, clang-tidy) only on changed files
4. **Dual linter overlap** — Ruff and Flake8 both run; Ruff is the modern path but Flake8 still catches some things via plugins
5. **Dual type checker** — Pyrefly replacing Mypy; both still configured

### How `spin lint` / `spin fixlint` Work

Defined in `.spin/cmds.py`. Under the hood:
```
uvx --python 3.10 lintrunner@0.12.7 [--take LINTERS] [--all-files | changed files]
```
- `spin lint` — runs fast linters on all files + slow linters on changed files only
- `spin fixlint` — same but applies auto-fixes (`--apply-patches`)
- `spin quicklint` / `spin quickfix` — changed files only, full linter set

### Ruff Rule Categories Enabled

B (bugbear), C4 (comprehensions), E/W (PEP 8), EXE, F (pyflakes), G (logging-format), LOG, NPY, PERF, PIE, PYI, Q, RSE, RUF, S101 (assert), SLOT, TC (type-checking), TRY, UP (pyupgrade), YTT, PT (pytest), PLC/PLE/PLR/PLW (pylint subset), SIM (simplify), FURB (refurb)

---

## Comparison With Other AI/ML Projects

### Summary Table

| Project | Linter | Formatter | Type Checker | Import Sort | Line Length | Orchestrator |
|---------|--------|-----------|-------------|-------------|-------------|-------------|
| **PyTorch** | Ruff + Flake8 | Ruff format | Pyrefly + Mypy | isort + usort | 88/120 | lintrunner + spin |
| **JAX** | Ruff (preview mode) | None | Mypy (+ Pyrefly manual/experimental) | Not configured | 80 | pre-commit |
| **TensorFlow** | pylint (custom rcfile) | None | None | None | 80 | Bazel CI scripts |
| **HuggingFace Transformers** | Ruff (`ALL` rules) | Ruff format | ty (Astral's new checker) | Ruff `I` | 119 | Makefile |
| **scikit-learn** | Ruff | Ruff format | Mypy | Ruff `I` | 88 | pre-commit |
| **LangChain** | Ruff (`ALL` rules) | Ruff format | Mypy (strict + pydantic plugin) | Ruff `I` | 88 | Makefile + pre-commit |
| **vLLM** | Ruff | Ruff format | Mypy (×4 Python versions) + pydantic | Ruff `I` | 88 | pre-commit (25 hooks) |

### Key Observations

1. **Ruff is near-universal** — 5 of 6 projects use Ruff for linting. TensorFlow is the only holdout (still on pylint). Most also use Ruff for formatting, replacing the old Flake8 + Black + isort triple.

2. **Type checker landscape is fragmenting** — Mypy remains most common (JAX, scikit-learn, LangChain, vLLM). HuggingFace Transformers has adopted **ty** (Astral's new type checker). PyTorch uses **Pyrefly** (Meta). JAX has Pyrefly as experimental/manual. vLLM has ty configured but not yet active.

3. **Line length varies by tradition** — 80 (JAX, TF — Google convention), 88 (scikit-learn, LangChain, vLLM — Black/Ruff default), 119-120 (HuggingFace, PyTorch Flake8).

4. **PyTorch has the most complex setup** — Dual linters (Ruff + Flake8), dual type checkers (Pyrefly + Mypy), ~55 linter definitions in lintrunner, performance tiering. This reflects its size, history, and mixed C++/Python codebase.

5. **No formatter is a real choice** — JAX and TensorFlow both lack an auto-formatter, relying on developer discipline and linter enforcement.

6. **pre-commit vs custom orchestration** — Most projects use pre-commit. PyTorch uses lintrunner (Meta's tool) for more control over execution order and performance. LangChain and HuggingFace use Makefiles.

7. **vLLM has the most elaborate pre-commit** — 25 hooks including multi-version mypy, typos (spell check), clang-format, license headers, Dockerfile validation, and custom scripts.

8. **Migration pattern** — Projects are consolidating: Flake8 + Black + isort → Ruff all-in-one. The type checker space is the next frontier of consolidation (Mypy → ty/Pyrefly).

---

## Recommendations for a New Project

### Minimal Modern Setup

**Tools:** Ruff (lint + format) + Mypy (types) + pre-commit (orchestration)

**`pyproject.toml`:**
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true

[tool.ruff.lint]
select = [
    "B",     # bugbear
    "C4",    # comprehensions
    "E", "W",# PEP 8
    "F",     # pyflakes
    "I",     # isort (built into Ruff)
    "UP",    # pyupgrade
    "SIM",   # simplify
    "RUF",   # ruff-specific
]

[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
disallow_untyped_defs = true
show_error_codes = true
warn_redundant_casts = true
```

**`.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        additional_dependencies: [...]
```

### When to Add More

- **codespell/typos** — if the project has user-facing docs or many contributors
- **clang-format/clang-tidy** — if the project has C/C++ extensions
- **ty** — watch Astral's type checker; HuggingFace Transformers is the early adopter signal
- **lintrunner** — if the project grows large enough that lint performance matters (PyTorch-scale)
- **Custom linters** — only when generic rules can't catch project-specific patterns

### Key Decisions to Make Early

1. **Line length** — 88 (Black default, most common) vs 80 (Google style) vs 119-120 (relaxed)
2. **Quote style** — double (most common, Ruff default) vs single
3. **Type checking strictness** — start strict (`disallow_untyped_defs`), relax per-module if needed
4. **Import sorting** — use Ruff's built-in `I` rule (replaces isort entirely)

---

## Source Files (PyTorch)

- `pyproject.toml` — Ruff, isort, usort config
- `.flake8` — Flake8 config
- `.lintrunner.toml` — All linter definitions
- `mypy.ini` — Mypy config
- `pyrefly.toml` — Pyrefly config
- `.spin/cmds.py` — spin lint/fixlint implementation
