# C Driver Bridge

This document explains how the C driver (`driver.c`) bridges Python and the GPU by loading compiled binaries and executing kernels via CUDA Driver API.

## Overview

The C driver bridges Python and the GPU by:
1. Loading compiled CUBIN into the GPU driver
2. Extracting arguments from Python objects
3. Calling CUDA Driver API to execute kernels

## Python Module Structure

**File:** `third_party/nvidia/backend/driver.c` (lines 1118-1144)

The C code is compiled as a Python extension module:
```c
static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS, "Load cubin into CUDA driver"},
    {"launch", launchKernel, METH_VARARGS, "launches cuda kernel"},
    {"fill_tma_descriptor", fillTMADescriptor, METH_VARARGS, "..."},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS, "..."},
    ...
};

// Module initialization
PyMODINIT_FUNC PyInit_cuda_utils(void) {
    ...
}
```

This creates `cuda_utils.cpython-*.so` which is imported in Python.

---

## Binary Loading: `loadBinary()`

**File:** `driver.c` (lines 156-218)

```c
static PyObject *loadBinary(PyObject *self, PyObject *args) {
    const char *name;       // Kernel function name
    const char *data;       // CUBIN binary bytes
    int shared;             // Shared memory size
    int device;             // GPU device ID

    PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared, &device);

    // 1. Ensure CUDA context exists
    CUcontext pctx = 0;
    cuCtxGetCurrent(&pctx);
    if (!pctx) {
        cuDevicePrimaryCtxRetain(&pctx, device);
        cuCtxSetCurrent(pctx);
    }

    // 2. Load CUBIN binary into GPU
    CUmodule mod;
    cuModuleLoadData(&mod, data);  // ← Key CUDA API call

    // 3. Get kernel function handle
    CUfunction fun;
    cuModuleGetFunction(&fun, mod, name);  // ← Key CUDA API call

    // 4. Query function attributes
    cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun);
    cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);
    cuFuncGetAttribute(&n_max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun);

    // 5. Configure dynamic shared memory (>48KB)
    if (shared > 49152) {
        cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED);
        cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...);
    }

    // Return (module_handle, function_handle, n_regs, n_spills, max_threads)
    return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills, n_max_threads);
}
```

**What this does:**
- Takes CUBIN bytes from Python
- Registers the binary with CUDA driver (`cuModuleLoadData`)
- Extracts the kernel function (`cuModuleGetFunction`)
- Returns handles for later execution

---

## Kernel Launch: `launchKernel()`

**File:** `driver.c` (lines 1005-1116)

```c
static PyObject *launchKernel(PyObject *self, PyObject *args) {
    // Parse arguments from Python
    int gridX, gridY, gridZ;
    uint64_t _stream, _function;
    int num_warps, num_ctas, shared_memory;
    PyObject *kernel_args;
    ...

    PyArg_ParseTuple(args, "iiiKKpp(iii)OOOOOOy*O",
        &gridX, &gridY, &gridZ,      // Grid dimensions
        &_stream, &_function,         // CUDA stream and function handles
        &launch_cooperative_grid,
        &launch_pdl,
        &num_warps, &num_ctas, &shared_memory,
        &launch_metadata,
        &launch_enter_hook, &launch_exit_hook,
        &global_scratch_obj, &profile_scratch_obj,
        &arg_annotations,
        &signature,                   // Type signature for argument extraction
        &kernel_args);                // Actual Python arguments

    // Call pre-launch hook (for profiling)
    launchHook(launch_enter_hook, launch_metadata);

    // Extract kernel arguments from Python objects
    void **params = (void **)alloca(num_params * sizeof(void *));
    for (int i = 0; i < num_args; ++i) {
        Extractor extractor = getExtractor(extractor_data[i]);
        params[i] = alloca(extractor.size);
        extractor.extract(params[i], args_data[i]);  // ← Convert Python → C
    }

    // Add scratch memory pointers
    extractPointer(params[params_idx++], global_scratch_obj);
    extractPointer(params[params_idx++], profile_scratch_obj);

    // Launch kernel (release GIL for concurrency)
    Py_BEGIN_ALLOW_THREADS;
    _launch(gridX, gridY, gridZ, num_warps, num_ctas, ..., params);
    Py_END_ALLOW_THREADS;

    // Call post-launch hook
    launchHook(launch_exit_hook, launch_metadata);

    Py_RETURN_NONE;
}
```

---

## Actual CUDA Launch: `_launch()`

**File:** `driver.c` (lines 570-636)

```c
static void _launch(int gridX, int gridY, int gridZ, int num_warps,
                    int num_ctas, int launch_cooperative_grid, int launch_pdl,
                    int shared_memory, CUstream stream, CUfunction function,
                    void **params) {

    // Build CUDA launch configuration
    CUlaunchConfig config;
    config.gridDimX = gridX * num_ctas;   // Total grid X
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;
    config.blockDimX = 32 * num_warps;    // Threads per block = warps × 32
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;

    // Add launch attributes
    CUlaunchAttribute launchAttr[4];
    int num_attrs = 0;

    // Cooperative grid (for multi-block synchronization)
    if (launch_cooperative_grid) {
        launchAttr[num_attrs++] = (CUlaunchAttribute){
            .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = 1
        };
    }

    // Cluster dimensions (Hopper feature)
    if (num_ctas != 1) {
        launchAttr[num_attrs++] = (CUlaunchAttribute){
            .id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
            .value.clusterDim = {num_ctas, 1, 1}
        };
    }

    config.attrs = launchAttr;
    config.numAttrs = num_attrs;

    // ★ Execute kernel on GPU ★
    cuLaunchKernelExHandle(&config, function, params, 0);
}
```

---

## Argument Extraction

**File:** `driver.c` (lines 640-779)

The C code extracts different types from Python objects:

```c
// Extract GPU device pointer from tensor
bool extractPointer(void *ptr, PyObject *obj) {
    if (PyLong_Check(obj)) {
        *dev_ptr = PyLong_AsUnsignedLongLong(obj);
        return true;
    }
    // Call tensor.data_ptr() method
    PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
    *dev_ptr = PyLong_AsUnsignedLongLong(ret);

    // Verify it's a valid GPU pointer
    cuPointerGetAttribute(dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
}

// Extract 32-bit integer
bool extractI32(void *ptr, PyObject *obj) {
    *((int32_t *)ptr) = PyLong_AsLong(obj);
    return PyErr_Occurred() == NULL;
}

// Extract 32-bit float
bool extractFP32(void *ptr, PyObject *obj) {
    double temp = PyFloat_AsDouble(obj);
    float f32 = (float)temp;
    *((uint32_t *)ptr) = *(uint32_t *)&f32;
}
```

**Extractor dispatch table:**
```c
Extractor extraction_map[] = {
    [EXTRACTOR_POINTER_INDEX] = {extractPointer, sizeof(CUdeviceptr)},
    [EXTRACTOR_INT32_INDEX] = {extractI32, sizeof(int32_t), .name = {"i1", "i32"}},
    [EXTRACTOR_FP32_INDEX] = {extractFP32, sizeof(uint32_t), .name = {"fp32"}},
    [EXTRACTOR_NVTMADESC_INDEX] = {extractTmaDesc, sizeof(CUtensorMap)},
    ...
};
```

---

## Python Launcher Class

**File:** `driver.py` (lines 277-316)

```python
class CudaLauncher(object):
    def __init__(self, src, metadata):
        # Get C launch function
        launcher = triton.runtime.driver.active.utils.launch

        # Build argument metadata
        self.arg_annotations = annotate_arguments(signature)
        self.kernel_signature = make_kernel_signature(signature)
        self.launch = launcher  # C function

    def __call__(self, gridX, gridY, gridZ, stream, function,
                 kernel_metadata, launch_metadata, enter_hook, exit_hook, *args):

        # Allocate scratch memory
        global_scratch = allocate_scratch(self.global_scratch_size, ...)
        profile_scratch = allocate_scratch(self.profile_scratch_size, ...)

        # Call C launch function
        self.launch(
            gridX, gridY, gridZ,
            stream, function,
            self.launch_cooperative_grid, self.launch_pdl,
            kernel_metadata, launch_metadata,
            enter_hook, exit_hook,
            global_scratch, profile_scratch,
            self.arg_annotations,
            self.kernel_signature,
            args  # Python arguments
        )
```

---

## Complete Call Flow

```
Python:
    kernel[(grid_x, grid_y)](x, y, N)
           ↓
    JITFunction.run()
           ↓
    kernel.run(gridX, gridY, gridZ, stream, function, ...)
           ↓
Python (driver.py):
    CudaLauncher.__call__(gridX, gridY, gridZ, stream, function, *args)
           ↓
C (driver.c):
    launchKernel(self, args)
        1. Parse Python arguments (PyArg_ParseTuple)
        2. Call launch_enter_hook
        3. Extract kernel arguments:
           - tensor.data_ptr() → CUdeviceptr
           - int → int32_t
           - float → float
        4. Allocate params array on stack
           ↓
    _launch(gridX, gridY, gridZ, ..., params)
        1. Build CUlaunchConfig
        2. Set launch attributes (cooperative, cluster)
           ↓
CUDA Driver API:
    cuLaunchKernelEx(&config, function, params, NULL)
           ↓
GPU:
    Kernel executes on device
           ↓
C:
    Return to Python after Py_END_ALLOW_THREADS
        5. Call launch_exit_hook
        6. Return None to Python
```

---

## Key CUDA Driver API Calls

| Function | Purpose | Location |
|----------|---------|----------|
| `cuModuleLoadData()` | Load CUBIN into GPU driver | `loadBinary()` line 182 |
| `cuModuleGetFunction()` | Get kernel function handle | `loadBinary()` line 184 |
| `cuFuncGetAttribute()` | Query register/spill count | `loadBinary()` lines 186-192 |
| `cuFuncSetAttribute()` | Configure shared memory | `loadBinary()` line 208 |
| `cuLaunchKernelEx()` | **Execute kernel on GPU** | `_launch()` line 634 |
| `cuPointerGetAttribute()` | Verify GPU pointer | `extractPointer()` line 669 |

---

## GIL Handling

The Global Interpreter Lock (GIL) is released during kernel launch:

```c
Py_BEGIN_ALLOW_THREADS;  // Release GIL
_launch(...);            // GPU execution can happen concurrently
Py_END_ALLOW_THREADS;    // Re-acquire GIL
```

This allows:
- Multiple Python threads to launch kernels concurrently
- Python code to run while GPU is executing
- Async GPU operations

---

## Next Steps

- [Backend Abstraction](05-backend-abstraction.md) - NVIDIA/AMD plugin architecture
