# 03 mlir compilation

This document provides a deep dive into Triton's MLIR-based compilation pipeline, covering the transformation from Python AST to GPU binary.

## Overview

```
Python AST → TTIR → TTGIR → LLIR → PTX → CUBIN
   (AST      (Triton  (TritonGPU  (LLVM   (PTX      (Binary)
   visitor)    IR)      IR)       IR)   assembly)
```

Each stage is an MLIR pass manager that transforms the IR.

***

## Stage 1: Python AST → TTIR (Triton IR)

**Entry Point:** `python/triton/compiler/code_generator.py` (line 1620)

```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
    # 1. Resolve argument types from signature
    arg_types = [None] * len(fn.arg_names)
    for k, v in src.signature.items():
        arg_types[idx] = str_to_ty(v, None)

    # 2. Create CodeGenerator (AST visitor)
    generator = CodeGenerator(context, prototype, ...)

    # 3. Visit Python AST → Generate MLIR operations
    generator.visit(fn.parse())

    # 4. Return MLIR module
    return generator.module
```

CodeGenerator (`code_generator.py:274`):

* Inherits from `ast.NodeVisitor`
* Visits Python AST nodes and emits MLIR operations
* Uses `TritonSemantic` (line 288) for operation semantics

Key visit methods:

* `visit_FunctionDef` - Creates MLIR function
* `visit_For` - Generates `scf.for` loops
* `visit_If` - Generates `scf.if` conditionals
* `visit_Call` (line 1420) - Maps `tl.*` calls to MLIR ops

Example transformation:

```python
# Python:
x = tl.load(ptr + offsets, mask=mask)

# Becomes MLIR (TTIR):
%1 = tt.splat %arg0 : ...
%2 = tt.load %1, %mask : tensor<128xf32>
```

***

## Stage 2: TTIR → TTGIR (TritonGPU IR)

**File:** `third_party/nvidia/backend/compiler.py` (lines 251-323)

This stage adds GPU-specific information:

* Layouts: How data is distributed across threads/warps
* Tensor Core ops: Matrix multiply-accumulate (MMA)
* Memory optimizations: Coalescing, prefetching

Key passes:

```python
def make_ttgir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)

    # Convert TTIR to TTGIR with layouts
    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)

    # Optimize memory access patterns
    passes.ttgpuir.add_coalesce(pm)                    # Coalesce memory accesses
    passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)        # Enable TensorCore for f32
    passes.ttgpuir.add_accelerate_matmul(pm)          # Use tensor cores for matmul
    passes.ttgpuir.add_optimize_dot_operands(pm)      # Optimize MMA operands

    # Loop optimizations
    passes.ttgpuir.add_fuse_nested_loops(pm)          # Fuse loops
    passes.ttgpuir.add_assign_latencies(pm)           # Assign instruction latencies
    passes.ttgpuir.add_schedule_loops(pm)             # Schedule for latency hiding
    passes.ttgpuir.add_pipeline(pm, opt.num_stages)   # Software pipelining

    # Memory prefetching
    passes.ttgpuir.add_prefetch(pm)                   # Add prefetch instructions
    passes.ttgpuir.add_coalesce_async_copy(pm)        # Coalesce async copies

    # Final cleanup
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    nvidia.passes.ttnvgpuir.add_fence_insertion(pm)   # Add memory fences

    pm.run(mod, 'make_ttgir')
```

Example transformation:

```mlir
// TTIR:
%1 = tt.dot %a, %b : tensor<128x64xf16>, tensor<64x128xf16>

// TTGIR (with layouts):
%1 = tt.dot %a, %b : tensor<128x64xf16, #mma>, tensor<64x128xf16, #mma>
```

***

## Stage 3: TTGIR → LLIR (LLVM IR)

**File:** `third_party/nvidia/backend/compiler.py` (lines 344-438)

This stage lowers TritonGPU IR to LLVM IR:

* Allocates shared memory
* Lowers tensor ops to PTX intrinsics
* Converts control flow

Key passes:

```python
def make_llir(self, src, metadata, options, capability):
    pm = ir.pass_manager(mod.context)

    # Memory allocation
    passes.convert.add_scf_to_cf(pm)                     # Structured CF to control flow
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm)  # Allocate shared memory
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)   # Allocate tensor memory
    passes.ttgpuir.add_allocate_global_scratch_memory(pm)    # Global scratch

    # Lower to LLVM
    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)        # NVGPU ops to LLVM
    passes.convert.add_nvvm_to_llvm(pm)                   # NVVM to LLVM

    pm.run(mod, 'make_llir')

    # Convert MLIR LLVM dialect to actual LLVM IR
    llvm.init_targets()
    context = llvm.context()
    llvm_mod = llvm.to_module(mod, context)

    # Link libdevice and optimize
    llvm.link_extern_libs(llvm_mod, paths)
    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    # Extract metadata
    metadata["shared"] = src.get_int_attr("ttg.shared")
    metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")

    return str(llvm_mod)
```

Shared memory allocation: The pass computes required shared memory and adds:

```mlir
module attributes {"ttg.shared" = 16384}  # 16KB shared memory
```

***

## Stage 4: LLIR → PTX

**File:** `third_party/nvidia/backend/compiler.py` (lines 440-464)

```python
def make_ptx(self, src, metadata, opt, capability):
    ptx_version = get_ptx_version_from_options(opt, self.target.arch)

    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)  # e.g., "sm_90"

    # Translate LLVM IR to PTX assembly
    ret = llvm.translate_to_asm(src, triple, proc, features, flags, opt.enable_fp_fusion, False)

    # Find kernel name from PTX
    names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
    metadata["name"] = names[0]

    # Fix up PTX version and target
    ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret)
    ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret)

    return ret  # PTX assembly as string
```

Example PTX output:

```ptx
.version 8.0
.target sm_90
.address_size 64

.visible .entry add_kernel(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 output_ptr,
    .param .u32 N
) {
    .reg .pred %p<2>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [x_ptr];
    ...
}
```

***

## Stage 5: PTX → CUBIN

**File:** `third_party/nvidia/backend/compiler.py` (lines 466-543)

```python
def make_cubin(self, src, metadata, opt, capability):
    ptxas = get_ptxas(self.target.arch).path

    # Write PTX to temp file
    with tempfile.NamedTemporaryFile(suffix='.ptx') as fsrc:
        fsrc.write(src)

        # Build ptxas command
        cmd = [
            ptxas,
            fsrc.name,
            '-o', fbin,
            f'--gpu-name=sm_{capability}',
            '-lineinfo',  # Debug info
        ]

        # Add max register constraint
        if opt.maxnreg:
            cmd.append(f'--maxrregcount={opt.maxnreg}')

        # Run ptxas
        subprocess.run(cmd, ...)

        # Read binary
        with open(fbin, 'rb') as f:
            cubin = f.read()

    return cubin  # Binary bytes
```

***

## MLIR Dialects Used

| Dialect         | Prefix | Purpose                    | Example Ops                     |
| --------------- | ------ | -------------------------- | ------------------------------- |
| Triton          | `tt`   | High-level tensor ops      | `tt.load`, `tt.store`, `tt.dot` |
| TritonGPU       | `ttg`  | GPU layouts, optimizations | Layout attributes               |
| TritonNvidiaGPU | `ttng` | NVIDIA-specific            | `ttng.warp_group_dot`, TMA      |
| SCF             | `scf`  | Structured control flow    | `scf.for`, `scf.if`             |
| LLVM            | `llvm` | LLVM IR in MLIR            | `llvm.call`, `llvm.load`        |
| NVVM            | `nvvm` | NVIDIA intrinsics          | `nvvm.read.ptx.sreg`            |

***

## Debugging the Pipeline

```bash
# Dump IR at every stage
MLIR_ENABLE_DUMP=1 python script.py

# Dump IR for specific kernel
MLIR_ENABLE_DUMP=my_kernel python script.py

# Dump to specific directory
MLIR_DUMP_PATH=/tmp/triton_ir python script.py

# Show MLIR pass timing
MLIR_ENABLE_TIMING=1 python script.py

# Override kernel at specific stage
TRITON_KERNEL_OVERRIDE=1 \
TRITON_OVERRIDE_DIR=/path/to/override \
python script.py
```

***

## Summary: From Python to GPU Binary

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

Transformation steps:

{% stepper %}
{% step %}
### AST → TTIR

* `tl.program_id(0)` → `tt.get_program_id(0)`
* `tl.arange(0, BLOCK_SIZE)` → `tt.make_range(0, 1024)`
* `tl.load(...)` → `tt.load %ptr, %mask`
{% endstep %}

{% step %}
### TTIR → TTGIR

* Add `#blocked` layout to tensors
* Insert `ttg.local_alloc` for shared memory
* Optimize memory access patterns
{% endstep %}

{% step %}
### TTGIR → LLIR

* Lower `tt.load` to `llvm.load` + address calculation
* Allocate shared memory (`llvm.inline_asm` for allocation)
* Insert barriers (`nvvm.barrier0`)
{% endstep %}

{% step %}
### LLIR → PTX

* LLVM backend generates PTX assembly
* Uses PTX-specific instructions (`ld.global`, `st.global`)
{% endstep %}

{% step %}
### PTX → CUBIN

* `ptxas` assembles PTX to machine code
* Output: ELF binary loadable by CUDA driver
{% endstep %}
{% endstepper %}

***

## Next Steps

* [C Driver Bridge](04-c-driver-bridge.md) - How Python calls GPU APIs
* [Backend Abstraction](05-backend-abstraction.md) - NVIDIA/AMD plugin architecture
