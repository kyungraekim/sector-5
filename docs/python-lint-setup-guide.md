# Python Lint & Format Setup Guide

A practical guide for setting up Ruff + Mypy + pre-commit on a new Python project.
Inspired by PyTorch's toolchain (see [python-lint-toolchain-research.md](python-lint-toolchain-research.md)).

Target: Python 3.12, dual line-length (format at 88, lint at 120).

---

## 1. Install Tools

```bash
# Ruff — linter + formatter
pip install ruff

# Mypy — type checker
pip install mypy

# pre-commit — git hook orchestrator
pip install pre-commit
```

Or add to your dev dependencies (e.g. in `pyproject.toml`):

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.14",
    "mypy>=1.15",
    "pre-commit>=4.0",
]
```

---

## 2. Configure Ruff

All Ruff config goes in `pyproject.toml`. The key idea borrowed from PyTorch: **the formatter
wraps at 88 (soft target), but the linter tolerates up to 120 (hard limit)**. This gives
developers breathing room for long lines that are more readable unsplit.

Add the following to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120            # hard lint limit (matches Flake8 B950 behavior)
target-version = "py312"

[tool.ruff.format]
line-length = 88             # soft format target — formatter wraps here
quote-style = "double"
docstring-code-format = true

[tool.ruff.lint]
select = [
    "B",      # flake8-bugbear — common pitfalls
    "C4",     # flake8-comprehensions — unnecessary list/dict calls
    "E", "W", # pycodestyle — PEP 8 basics
    "F",      # pyflakes — undefined names, unused imports
    "G",      # flake8-logging-format — logging best practices
    "I",      # isort — import sorting (built into Ruff)
    "PERF",   # perflint — performance anti-patterns
    "PT",     # flake8-pytest-style — pytest best practices
    "RUF",    # ruff-specific rules
    "SIM",    # flake8-simplify — simplifiable constructs
    "UP",     # pyupgrade — modernize syntax for target Python
]
ignore = [
    "E501",   # line too long — handled by formatter + line-length=120 above
]

[tool.ruff.lint.isort]
combine-as-imports = true
lines-after-imports = 2
```

### Why these rules?

| Rule | What it catches | Example |
|------|----------------|---------|
| B | Mutable default args, bare `except`, `setattr` with constants | `def f(x=[]):` |
| C4 | `list(x for x in ...)` → `[x for x in ...]` | Unnecessary wrappers |
| E, W | Basic PEP 8 (whitespace, indentation) | Trailing whitespace |
| F | Unused imports, undefined names | `import os` (unused) |
| G | `logging.info(f"...")` → `logging.info("...", ...)` | f-string in logging |
| I | Unsorted imports | Enforces consistent import order |
| PERF | `dict()` in hot loop, unnecessary list copies | `list(x)` when iterable suffices |
| PT | `assertEqual` → `assert x ==` in pytest, fixture issues | pytest style |
| RUF | Ruff-specific: unnecessary `noqa`, mutable class attrs | Meta-linting |
| SIM | `if x: return True else: return False` → `return x` | Simplifiable logic |
| UP | `Dict[str, int]` → `dict[str, int]` for Python 3.12 | Old-style annotations |

### Rules to add later as the codebase matures

```toml
# Stricter — enable when team is ready
# "TRY",    # tryceratops — exception handling patterns
# "TC",     # flake8-type-checking — TYPE_CHECKING block optimization
# "PIE",    # flake8-pie — misc. lint
# "FURB",   # refurb — modernization suggestions
# "S",      # bandit/security (start with "S101" to ban assert in prod code)
# "PLC", "PLE", "PLR", "PLW",  # pylint subset
```

---

## 3. Configure Mypy

Add to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
check_untyped_defs = true
disallow_untyped_defs = true
disallow_untyped_decorators = true
warn_redundant_casts = true
warn_return_any = true
show_error_codes = true
```

### Per-module overrides

For modules with heavy dynamic typing or third-party dependencies that lack stubs:

```toml
[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false        # tests don't need full annotations

[[tool.mypy.overrides]]
module = ["some_third_party_lib.*"]
ignore_missing_imports = true        # no stubs available
```

PyTorch uses this pattern extensively — strict globally, relaxed per-module.

---

## 4. Configure pre-commit

Create `.pre-commit-config.yaml` in the project root:

```yaml
repos:
  # Ruff — lint + format
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
        args: [--line-length=88]   # soft format target

  # Mypy — type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        # List type stubs your project needs:
        additional_dependencies: []
        # Optionally restrict to source dirs:
        # args: [--config-file=pyproject.toml]

  # Optional: codespell — catch typos in comments/strings
  # - repo: https://github.com/codespell-project/codespell
  #   rev: v2.4.1
  #   hooks:
  #     - id: codespell
  #       args: [--ignore-words-list, "some,words,to,skip"]
```

Install the hooks:

```bash
pre-commit install
```

Now linting runs automatically on every `git commit`.

---

## 5. CI Integration

Add a GitHub Actions workflow. This ensures the lint check runs even if a developer
skips pre-commit locally.

Create `.github/workflows/lint.yml`:

```yaml
name: Lint

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install ruff mypy

      - name: Ruff lint
        run: ruff check .

      - name: Ruff format check
        run: ruff format --check --line-length=88 .

      - name: Mypy
        run: mypy .
```

---

## 6. Makefile Shortcuts

For developers who prefer explicit commands over pre-commit hooks:

```makefile
.PHONY: lint fix typecheck

lint:                          ## Run all lint checks
	ruff check .
	ruff format --check --line-length=88 .
	mypy .

fix:                           ## Auto-fix lint issues and format
	ruff check --fix .
	ruff format --line-length=88 .

typecheck:                     ## Run type checker only
	mypy .
```

---

## 7. Editor Integration

### VS Code (`.vscode/settings.json`)

```json
{
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        },
        "editor.rulers": [88, 120]
    },
    "mypy.runUsingActiveInterpreter": true
}
```

The two rulers give a visual cue: 88 is where the formatter wraps, 120 is the hard limit.

### PyCharm / IntelliJ

1. Install the **Ruff** plugin (Settings → Plugins → Marketplace)
2. Enable "Format on Save" and "Fix on Save" in Settings → Tools → Ruff
3. Set hard wrap at 120, visual guide at 88 (Settings → Editor → Code Style → Python)

---

## 8. Onboarding Checklist

Share this with new team members:

```
1. Clone the repo
2. pip install -e ".[dev]"       # installs ruff, mypy, pre-commit
3. pre-commit install            # activates git hooks
4. (Optional) Install the Ruff extension in your editor
5. Run `make lint` to verify your setup works
```

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `ruff check .` | Lint all files |
| `ruff check --fix .` | Lint + auto-fix |
| `ruff format .` | Format all files (wraps at 88) |
| `ruff format --check .` | Check formatting without modifying |
| `mypy .` | Type check all files |
| `pre-commit run --all-files` | Run all hooks on entire repo |
| `make lint` | Run all checks (via Makefile) |
| `make fix` | Auto-fix + format (via Makefile) |

---

## Dual Line-Length Explained

This setup uses a pattern borrowed from PyTorch:

```
 0          88         120
 |──────────|───────────|
 │  formatted zone      │
 │          │  allowed  │  ← lint error
 │          │  overflow │
```

- **Ruff format** wraps lines at **88** characters (the standard Black default).
- **Ruff lint** (with E501 ignored, line-length=120) only flags lines exceeding **120**.
- Lines between 88–120 are left alone by the formatter and pass lint. This is useful for
  long strings, URLs, complex expressions, or deeply nested code that reads better unsplit.

This avoids the common frustration of the formatter aggressively wrapping lines that
were deliberately kept long for readability.
