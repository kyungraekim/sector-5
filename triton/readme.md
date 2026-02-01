# README

This documentation explains how Triton exposes GPU binaries as Python APIs, covering the complete flow from `@triton.jit` decorated functions to GPU execution.

## Overview

Triton uses a JIT (Just-In-Time) compilation approach where Python-decorated kernel functions are compiled to GPU binaries at runtime and executed via CUDA/HIP driver APIs. The key insight is that **Python never directly executes GPU code** - instead, Python orchestrates compilation and hands off execution to C code that calls native GPU driver APIs.

## Documentation Structure

| Document                                               | Description                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| [01-overview](01-overview.md)                          | High-level architecture summary with key components and file references        |
| [02-jit-and-caching.md](02-jit-and-caching.md)         | Deep dive into the JIT system, caching mechanisms, and argument specialization |
| [03-mlir-compilation.md](03-mlir-compilation.md)       | MLIR compilation pipeline from Python AST to GPU binary                        |
| [04-c-driver-bridge.md](04-c-driver-bridge.md)         | C driver implementation and CUDA API calls                                     |
| [05-backend-abstraction.md](05-backend-abstraction.md) | Backend plugin architecture for NVIDIA/AMD support                             |

## Quick Reference

### The Complete Flow

{% stepper %}
{% step %}
### Python code (user)

@triton.jit decorated functions are defined in Python. Python is used to express kernels but does not execute GPU code.
{% endstep %}

{% step %}
### JITFunction wrapper

A Python wrapper that caches compiled kernels, manages compilation triggers, and handles argument specialization.
{% endstep %}

{% step %}
### Compilation Pipeline

Transforms the Python AST/IR through internal representations:

* AST → TTIR → TTGIR → LLIR → PTX → CUBIN
{% endstep %}

{% step %}
### CompiledKernel

Holds the generated GPU binary and associated metadata (launch parameters, symbol info).
{% endstep %}

{% step %}
### CudaLauncher

A Python class that prepares launch parameters and calls into the C driver bridge for execution.
{% endstep %}

{% step %}
### C Driver

C code (e.g., `driver.c`) invokes the CUDA Driver API (or HIP) to launch the kernel on the GPU.
{% endstep %}
{% endstepper %}

### Key Files

| Component           | File                                     |
| ------------------- | ---------------------------------------- |
| JIT decorator       | `python/triton/runtime/jit.py`           |
| Compiler            | `python/triton/compiler/compiler.py`     |
| NVIDIA backend      | `third_party/nvidia/backend/compiler.py` |
| C driver            | `third_party/nvidia/backend/driver.c`    |
| Backend abstraction | `python/triton/backends/`                |

## Getting Started

Start with [01-overview.md](01-overview.md) for a high-level understanding, then dive into specific topics based on your interest:

* **Understanding the JIT system**: Read [02-jit-and-caching.md](02-jit-and-caching.md)
* **Understanding compilation**: Read [03-mlir-compilation.md](03-mlir-compilation.md)
* **Understanding GPU execution**: Read [04-c-driver-bridge.md](04-c-driver-bridge.md)
* **Adding a new backend**: Read [05-backend-abstraction.md](05-backend-abstraction.md)
