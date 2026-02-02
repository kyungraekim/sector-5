# Triton Architecture Overview

## How GPU Binaries Are Exposed as Python APIs

Triton bridges Python and GPU execution through a sophisticated JIT compilation system. This document provides a high-level overview of the architecture.

## Architecture Summary

```
Python Code (@triton.jit decorated)
       ↓
JITFunction wrapper (caches, manages compilation)
       ↓
Compilation Pipeline (AST → TTIR → TTGIR → LLIR → PTX → CUBIN)
       ↓
CompiledKernel (holds binary + metadata)
       ↓
CudaLauncher (Python class calling C code)
       ↓
C Driver (driver.c) → CUDA Driver API → GPU Execution
```

## Key Components

### 1. Python JIT Decorator (`@triton.jit`)

**File:** `python/triton/runtime/jit.py` (lines 921-973)

- Wraps user kernel functions in `JITFunction` class
- Enables `kernel[grid](*args)` syntax via `__getitem__` method (line 364-370)
- Manages per-device kernel caching

### 2. Compilation Pipeline

**File:** `python/triton/compiler/compiler.py` (lines 226-363)

Stages (for NVIDIA):
1. **TTIR** - Triton IR (from Python AST via MLIR)
2. **TTGIR** - TritonGPU IR (GPU-specific layouts, optimizations)
3. **LLIR** - LLVM IR
4. **PTX** - NVIDIA PTX assembly
5. **CUBIN** - Binary (via `ptxas` assembler)

### 3. CompiledKernel

**File:** `python/triton/compiler/compiler.py` (lines 407-504)

- Holds the compiled CUBIN binary
- Stores metadata (num_warps, shared_memory, registers)
- Provides `run()` method to launch kernel

### 4. CUDA Launcher

**File:** `third_party/nvidia/backend/driver.py` (lines 277-316)

- Python class that bridges to C code
- Handles argument extraction from PyTorch tensors
- Calls C `launch()` function

### 5. C Driver (The Critical Bridge)

**File:** `third_party/nvidia/backend/driver.c`

This is where Python meets GPU:

**Binary Loading (lines 156-218):**
```c
cuModuleLoadData(&mod, cubin_data);         // Load CUBIN into GPU
cuModuleGetFunction(&fun, mod, "kernel");   // Get kernel function handle
```

**Kernel Launch (lines 570-636):**
```c
cuLaunchKernelEx(&config, function, params, NULL);  // Execute on GPU
```

## The Python-to-GPU Bridge (How It Actually Works)

### Step 1: User calls kernel
```python
add_kernel[(n_elements,)](x, y, output, n_elements, BLOCK_SIZE=1024)
```

### Step 2: JITFunction.run() (jit.py:708-763)
- Gets current device/stream
- Computes cache key from argument types
- Checks if kernel already compiled
- If not cached, triggers compilation

### Step 3: Compilation (compiler.py:226-363)
- Backend (CUDA/AMD) defines compilation stages
- Each stage transforms IR
- Final stage produces binary (CUBIN/HSACO)
- Binary cached to `~/.triton/cache/`

### Step 4: CompiledKernel initialization (compiler.py:439-474)
- Lazy initialization on first launch
- Calls `load_binary()` C function
- C code calls `cuModuleLoadData()` to register binary with GPU driver
- Returns function handle for later launches

### Step 5: Kernel execution (driver.c:570-636)
- Arguments extracted from Python objects (tensor.data_ptr())
- Build CUDA launch configuration (grid, block, shared memory)
- Call `cuLaunchKernelEx()` to execute on GPU
- GPU runs the kernel asynchronously

## Key File Reference

| Component | File | Lines |
|-----------|------|-------|
| `@triton.jit` decorator | `python/triton/runtime/jit.py` | 921-973 |
| `JITFunction` class | `python/triton/runtime/jit.py` | 610-803 |
| `run()` method | `python/triton/runtime/jit.py` | 708-763 |
| `compile()` function | `python/triton/compiler/compiler.py` | 226-363 |
| `CompiledKernel` class | `python/triton/compiler/compiler.py` | 407-504 |
| NVIDIA backend stages | `third_party/nvidia/backend/compiler.py` | 545-556 |
| `CudaLauncher` | `third_party/nvidia/backend/driver.py` | 277-316 |
| `loadBinary()` C function | `third_party/nvidia/backend/driver.c` | 156-218 |
| `_launch()` C function | `third_party/nvidia/backend/driver.c` | 570-636 |
| Driver selection | `python/triton/runtime/driver.py` | 8-49 |

## Critical CUDA API Calls (in driver.c)

1. **cuModuleLoadData()** - Loads compiled CUBIN binary into GPU driver
2. **cuModuleGetFunction()** - Gets executable kernel function handle
3. **cuLaunchKernelEx()** - Executes kernel on GPU with specified grid/block/args

These are the actual CUDA Driver API calls that make GPU execution happen. The rest of Triton (Python frontend, MLIR compilation, caching) exists to:
1. Transform Python code into GPU binary
2. Cache compiled kernels for reuse
3. Marshal Python arguments to C-compatible format
4. Call these CUDA APIs with the right parameters

## Summary

The key insight is that **no Python code runs on the GPU**. Instead:

1. Python decorators capture kernel source code
2. MLIR-based compiler transforms Python AST → GPU binary
3. C code loads binary into GPU via CUDA Driver API
4. C code extracts arguments from Python objects (tensor pointers, scalars)
5. C code calls `cuLaunchKernelEx()` to execute the kernel
6. Results are written directly to GPU memory (in-place on tensors)

This architecture allows Triton to provide a Python-first developer experience while achieving near-native GPU performance.

## Next Steps

- [JIT and Caching](02-jit-and-caching.md) - Deep dive into the JIT system
- [MLIR Compilation](03-mlir-compilation.md) - Understanding the compilation pipeline
- [C Driver Bridge](04-c-driver-bridge.md) - How Python calls GPU APIs
- [Backend Abstraction](05-backend-abstraction.md) - NVIDIA/AMD plugin architecture
