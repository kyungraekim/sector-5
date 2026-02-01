# 05 backend abstraction

This document explains Triton's plugin architecture for GPU backends, covering backend discovery, the key abstractions, and how to add a new backend.

## Overview

Triton uses a plugin architecture for GPU backends. NVIDIA and AMD are built-in, but the system supports external backends via Python entry points.

## Backend Discovery

**File:** `python/triton/backends/__init__.py` (lines 38-66)

```python
def _discover_backends() -> dict[str, Backend]:
    backends = dict()

    # Option 1: Fast path - scan third_party/ directory
    if os.environ.get("TRITON_BACKENDS_IN_TREE", "") == "1":
        for name in os.listdir(root):
            compiler = importlib.import_module(f"triton.backends.{name}.compiler")
            driver = importlib.import_module(f"triton.backends.{name}.driver")
            backends[name] = Backend(
                _find_concrete_subclasses(compiler, BaseBackend),
                _find_concrete_subclasses(driver, DriverBase)
            )

    # Option 2: Entry points for external plugins
    else:
        for ep in entry_points().select(group="triton.backends"):
            compiler = importlib.import_module(f"{ep.value}.compiler")
            driver = importlib.import_module(f"{ep.value}.driver")
            backends[ep.name] = Backend(compiler_class, driver_class)

    return backends

# Global backends registry
backends: dict[str, Backend] = _discover_backends()
```

***

## Key Abstractions

### GPUTarget

**File:** `backends/compiler.py` (lines 8-14)

```python
@dataclass(frozen=True)
class GPUTarget:
    backend: str      # "cuda", "hip"
    arch: int | str   # 90 (NVIDIA CC), "gfx940" (AMD)
    warp_size: int    # 32 (NVIDIA), 64 (AMD)
```

### BaseBackend

**File:** `backends/compiler.py` (lines 23-92)

```python
class BaseBackend(metaclass=ABCMeta):
    def __init__(self, target: GPUTarget):
        self.target = target

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget) -> bool:
        """Check if this backend supports the given target."""

    @abstractmethod
    def hash(self) -> str:
        """Unique ID for cache invalidation."""

    @abstractmethod
    def parse_options(self, options: dict) -> object:
        """Parse and validate compiler options."""

    @abstractmethod
    def add_stages(self, stages: dict, options: object):
        """Add compilation stages to pipeline."""

    @abstractmethod
    def load_dialects(self, context):
        """Load MLIR dialects for this backend."""

    @abstractmethod
    def get_module_map(self) -> Dict[str, ModuleType]:
        """Map interface modules to implementations."""
```

### DriverBase

**File:** `backends/driver.py` (lines 11-47)

```python
class DriverBase(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def is_active(self) -> bool:
        """Is this driver available on the system?"""

    @abstractmethod
    def get_current_target(self) -> GPUTarget:
        """Get target for current device."""

    @abstractmethod
    def map_python_to_cpp_type(self, ty: str) -> str:
        """Map Triton types to C++ types."""

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """Return benchmarking function."""
```

***

## Driver Selection

**File:** `python/triton/runtime/driver.py`

```python
def _create_driver() -> DriverBase:
    # Option 1: User-specified backend
    selected = os.environ.get("TRITON_DEFAULT_BACKEND", None)
    if selected:
        return backends[selected].driver()

    # Option 2: Auto-detect active driver
    active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(active_drivers) != 1:
        raise RuntimeError(f"{len(active_drivers)} active drivers")
    return active_drivers[0]()

# Global driver singleton
driver = DriverConfig()

# Access active driver
driver.active  # Returns CudaDriver or HipDriver
```

***

## NVIDIA Backend Implementation

### Compiler

**File:** `third_party/nvidia/backend/compiler.py`

```python
class CUDABackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "cuda"

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"] = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["llir"] = lambda src, meta: self.make_llir(src, meta, options)
        stages["ptx"] = lambda src, meta: self.make_ptx(src, meta, options)
        stages["cubin"] = lambda src, meta: self.make_cubin(src, meta, options)
```

### Driver

**File:** `third_party/nvidia/backend/driver.py`

```python
class CudaDriver(GPUDriver):
    def __init__(self):
        self.utils = CudaUtils()      # C bindings (cuda_utils.so)
        self.launcher_cls = CudaLauncher

    @staticmethod
    def is_active():
        import torch
        return torch.cuda.is_available() and (torch.version.hip is None)

    def get_current_target(self):
        capability = torch.cuda.get_device_capability()
        return GPUTarget("cuda", capability[0]*10 + capability[1], 32)
```

***

## AMD Backend Implementation

### Compiler

**File:** `third_party/amd/backend/compiler.py`

```python
class AMDBackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "hip"

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"] = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["llir"] = lambda src, meta: self.make_llir(src, meta, options)
        stages["amdgcn"] = lambda src, meta: self.make_amdgcn(src, meta, options)
        stages["hsaco"] = lambda src, meta: self.make_hsaco(src, meta, options)
```

### Driver

**File:** `third_party/amd/backend/driver.py`

```python
class HIPDriver(GPUDriver):
    @staticmethod
    def is_active():
        import torch
        return torch.cuda.is_available() and (torch.version.hip is not None)

    def get_current_target(self):
        props = torch.cuda.get_device_properties(device)
        arch = props.gcnArchName  # e.g., "gfx90a"
        return GPUTarget("hip", arch, 64)  # AMD uses 64-thread warps
```

***

## How Backends Are Selected at Runtime

{% stepper %}
{% step %}
### Kernel launch triggers JIT

User calls kernel:

```python
kernel[grid](x, y, output)
```
{% endstep %}

{% step %}
### Get current target

JITFunction.run() gets current target:

```python
target = driver.active.get_current_target()  # GPUTarget("cuda", 90, 32)
```
{% endstep %}

{% step %}
### Create backend for target

Get appropriate backend:

```python
backend = make_backend(target)  # Returns CUDABackend(target)
```
{% endstep %}

{% step %}
### Backend defines compilation stages

Backend populates stages:

```python
stages = {}
backend.add_stages(stages, options, Language.TRITON)
```
{% endstep %}

{% step %}
### Compile through stages

Pipeline executes stages in order:

```python
for stage_name, stage_fn in stages.items():
    module = stage_fn(module, metadata)
```
{% endstep %}
{% endstepper %}

***

## Adding a New Backend

{% stepper %}
{% step %}
### Create directory structure

```
third_party/<backend>/
├── backend/
│   ├── __init__.py
│   ├── compiler.py    # Implements BaseBackend
│   └── driver.py      # Implements DriverBase
├── lib/               # C++/MLIR code
└── include/           # Headers
```
{% endstep %}

{% step %}
### Implement BaseBackend

```python
class MyBackend(BaseBackend):
    @staticmethod
    def supports_target(target):
        return target.backend == "my_backend"

    def add_stages(self, stages, options, language):
        stages["ttir"] = self.make_ttir
        stages["my_ir"] = self.make_my_ir
        stages["binary"] = self.make_binary
```
{% endstep %}

{% step %}
### Implement DriverBase

```python
class MyDriver(GPUDriver):
    @staticmethod
    def is_active():
        return my_gpu_available()

    def get_current_target(self):
        return GPUTarget("my_backend", arch, warp_size)
```
{% endstep %}

{% step %}
### Register via entry point (pyproject.toml)

```toml
[project.entry-points."triton.backends"]
my_backend = "triton.backends.my_backend"
```
{% endstep %}
{% endstepper %}

***

## Backend Selection Environment Variables

| Variable                  | Purpose                    | Example       |
| ------------------------- | -------------------------- | ------------- |
| `TRITON_DEFAULT_BACKEND`  | Force specific backend     | `cuda`, `hip` |
| `TRITON_BACKENDS_IN_TREE` | Skip entry point discovery | `1`           |

***

## Compilation Stage Differences

| Stage     | NVIDIA              | AMD                 |
| --------- | ------------------- | ------------------- |
| IR        | TTIR → TTGIR → LLIR | TTIR → TTGIR → LLIR |
| Assembly  | PTX                 | AMDGCN              |
| Binary    | CUBIN (via ptxas)   | HSACO (via lld)     |
| Assembler | ptxas               | lld                 |

***

## Summary

The backend abstraction enables:

* Plugin architecture: New backends can be added without modifying core Triton
* Runtime detection: Automatically selects available GPU
* Unified API: Same Python API works across NVIDIA/AMD
* Customizable pipeline: Each backend defines its own compilation stages
