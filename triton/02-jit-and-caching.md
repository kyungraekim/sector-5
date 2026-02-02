# JIT System and Caching

This document provides a deep dive into Triton's JIT compilation system, including the `@triton.jit` decorator, caching mechanisms, and argument specialization.

## The @triton.jit Decorator

**File:** `python/triton/runtime/jit.py` (lines 921-973)

When you write `@triton.jit`, the decorator:
1. Wraps your function in a `JITFunction` object
2. Extracts the source code via `inspect.getsourcelines()` (line 462)
3. Stores function metadata: signature, parameters, constexprs
4. Creates a per-device cache for compiled kernels (line 790)

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
# Result: add_kernel is now a JITFunction object, not a regular function
```

## JITFunction Class Hierarchy

```
JITCallable (lines 456-561)
    ├── Stores source code and hash
    ├── Manages dependency tracking
    └── Computes cache key from source

JITFunction (lines 610-803)
    ├── Extends JITCallable
    ├── Per-device kernel cache (defaultdict)
    ├── Argument specialization
    └── run() method for execution
```

## The `kernel[grid](*args)` Syntax

**How it works:** `JITFunction.__getitem__` (line 364-370):
```python
def __getitem__(self, grid) -> T:
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```
This returns a lambda that captures the grid and calls `run()`.

---

## Cache Key Computation (Multi-Level)

### Level 1: In-Memory Cache (Per-Device)

**File:** `python/triton/runtime/jit.py` (lines 574-595)

```python
def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    # Returns string combining specialization + options
```

**What's in the specialization?**
- Argument types (e.g., `"*fp32"` for float pointer)
- Constexpr values (compiled into the kernel)
- Specialization attributes (e.g., divisibility by 16)

**Where it's stored:**
```python
self.device_caches = defaultdict(self.create_binder)  # line 790
# Structure: {device_id: (kernel_cache, kernel_key_cache, target, backend, binder)}
```

### Level 2: Disk Cache (Across Sessions)

**File:** `python/triton/compiler/compiler.py` (lines 245-279)

```python
# Compute hash from source + options + backend
key = get_cache_key(src, backend, options, env_vars=env_vars)
hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
fn_cache_manager = get_cache_manager(hash)

# Check if already compiled
metadata_path = metadata_group.get(metadata_filename)
if not always_compile and metadata_path is not None:
    # Cache hit!
    return CompiledKernel(src, metadata_group, hash)
```

**Cache location:** `~/.triton/cache/<hash>/`
```
<hash>/
├── kernel_name.json      # Metadata (num_warps, shared_mem, etc.)
├── kernel_name.ttir      # Triton IR
├── kernel_name.ttgir     # TritonGPU IR
├── kernel_name.llir      # LLVM IR
├── kernel_name.ptx       # PTX assembly
└── kernel_name.cubin     # Binary
```

---

## Argument Specialization

**File:** `python/triton/runtime/jit.py` (lines 392-449)

Triton generates a specialized "binder" function for each kernel:

```python
def create_function_from_signature(sig, kparams, backend):
    # Dynamically generates a Python function like:
    # def dynamic_func(x_ptr, y_ptr, N, **options):
    #     params = {'x_ptr': x_ptr, 'y_ptr': y_ptr, 'N': N}
    #     specialization = [
    #         specialize_impl(backend, x_ptr, False, True, True),
    #         specialize_impl(backend, y_ptr, False, True, True),
    #         ("constexpr", N)  # if N is constexpr
    #     ]
    #     return params, specialization, options
```

**What specialization captures:**
1. **Type:** `"*fp32"`, `"*fp16"`, `"i32"`, etc.
2. **Alignment:** Is the pointer 16-byte aligned?
3. **Divisibility:** Is the integer divisible by 16?
4. **Constexpr:** Compile-time constant value

**Why this matters:**
- Aligned loads/stores are faster on GPU
- Known divisibility enables loop unrolling
- Constexpr values are baked into the binary

---

## The run() Method - Complete Flow

**File:** `python/triton/runtime/jit.py` (lines 708-763)

```python
def run(self, *args, grid, warmup, **kwargs):
    # 1. Get device info
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)

    # 2. Get per-device cache
    kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]

    # 3. Specialize arguments (binder was pre-computed)
    bound_args, specialization, options = binder(*args, **kwargs)

    # 4. Compute cache key
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(key, None)

    # 5. Compile if not cached
    if kernel is None:
        kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)

    # 6. Launch kernel
    if not warmup:
        grid = grid(bound_args) if callable(grid) else grid
        kernel.run(grid_0, grid_1, grid_2, stream, ...)

    return kernel
```

---

## Compilation Trigger (_do_compile)

**File:** `python/triton/runtime/jit.py` (lines 859-888)

```python
def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
    # 1. Create ASTSource (wraps kernel + metadata)
    src = self.ASTSource(self, signature, constexprs, attrs)

    # 2. Call compile() - this is the actual compilation
    kernel = self.compile(src, target=target, options=options.__dict__)

    # 3. Store in memory cache
    kernel_cache[key] = kernel

    return kernel
```

---

## Cache Invalidation

Triton tracks when cache should be invalidated:

1. **Source code change:** Hash of function source (line 471)
2. **Global variable change:** Tracked at runtime (lines 745-748)
3. **Option change:** Different num_warps, num_stages, etc.
4. **Target change:** Different GPU architecture
5. **Environment variables:** `TRITON_ALWAYS_COMPILE=1` bypasses cache

```python
# Check that used global values have not changed
for (name, _), (val, globals_dict) in self.used_global_vals.items():
    if (newVal := globals_dict.get(name, not_present)) != val:
        raise RuntimeError(f"Global variable {name} has changed...")
```

---

## Environment Variables for Caching

| Variable | Purpose |
|----------|---------|
| `TRITON_CACHE_DIR` | Custom cache location |
| `TRITON_ALWAYS_COMPILE=1` | Bypass cache, always recompile |
| `TRITON_STORE_BINARY_ONLY=1` | Only cache final binary |
| `TRITON_CACHE_AUTOTUNING=1` | Cache autotune results |

---

## Next Steps

- [MLIR Compilation](03-mlir-compilation.md) - Understanding the compilation pipeline
- [C Driver Bridge](04-c-driver-bridge.md) - How Python calls GPU APIs
