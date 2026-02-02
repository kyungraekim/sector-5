# Triton Architecture Documentation

This documentation explains how Triton exposes GPU binaries as Python APIs, covering the complete flow from `@triton.jit` decorated functions to GPU execution.

## Overview

Triton uses a JIT (Just-In-Time) compilation approach where Python-decorated kernel functions are compiled to GPU binaries at runtime and executed via CUDA/HIP driver APIs. The key insight is that **Python never directly executes GPU code** - instead, Python orchestrates compilation and hands off execution to C code that calls native GPU driver APIs.

## Documentation Structure

| Document | Description |
|----------|-------------|
| [01-overview.md](01-overview.md) | High-level architecture summary with key components and file references |
| [02-jit-and-caching.md](02-jit-and-caching.md) | Deep dive into the JIT system, caching mechanisms, and argument specialization |
| [03-mlir-compilation.md](03-mlir-compilation.md) | MLIR compilation pipeline from Python AST to GPU binary |
| [04-c-driver-bridge.md](04-c-driver-bridge.md) | C driver implementation and CUDA API calls |
| [05-backend-abstraction.md](05-backend-abstraction.md) | Backend plugin architecture for NVIDIA/AMD support |

## Quick Reference

### The Complete Flow

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

### Key Files

| Component | File |
|-----------|------|
| JIT decorator | `python/triton/runtime/jit.py` |
| Compiler | `python/triton/compiler/compiler.py` |
| NVIDIA backend | `third_party/nvidia/backend/compiler.py` |
| C driver | `third_party/nvidia/backend/driver.c` |
| Backend abstraction | `python/triton/backends/` |

## Getting Started

Start with [01-overview.md](01-overview.md) for a high-level understanding, then dive into specific topics based on your interest:

- **Understanding the JIT system**: Read [02-jit-and-caching.md](02-jit-and-caching.md)
- **Understanding compilation**: Read [03-mlir-compilation.md](03-mlir-compilation.md)
- **Understanding GPU execution**: Read [04-c-driver-bridge.md](04-c-driver-bridge.md)
- **Adding a new backend**: Read [05-backend-abstraction.md](05-backend-abstraction.md)
