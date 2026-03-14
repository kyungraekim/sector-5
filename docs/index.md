# Sector-5

Deep technical documentation on GPU computing infrastructure — from low-level kernel compilation to large-scale LLM training optimizations.

---

## Topics

<div class="grid cards" markdown>

-   **Triton Internals**

    ---

    How Triton compiles `@triton.jit` Python functions into GPU binaries and executes them via CUDA/HIP driver APIs. Covers the full pipeline from Python AST to PTX to GPU execution.

    - JIT compilation and kernel caching
    - MLIR-based compilation pipeline (TTIR → TTGIR → LLIR → PTX → CUBIN)
    - C driver bridge and CUDA API calls
    - Backend plugin architecture (NVIDIA/AMD)

    [:octicons-arrow-right-24: Triton docs](triton/01-overview.md)

-   **Megatron-LM: GPU Memory**

    ---

    Where GPU memory goes during LLM training, how to estimate it before launching a run, and strategies to prevent OOM errors — including why OOM can appear only after many training steps.

    - Memory breakdown: parameters, gradients, optimizer states, activations
    - Parallelism effects on memory footprint
    - Selective activation recomputation (MLP checkpointing)
    - Runtime monitoring and reduction strategies

    [:octicons-arrow-right-24: GPU memory docs](megatron/gpu_memory/gpu_memory_guide.md)

-   **Megatron-LM: Knowledge Distillation**

    ---

    How Megatron-LM and NVIDIA ModelOpt implement knowledge distillation — transferring learned representations from a large teacher to a smaller student model.

    - Two-library architecture (Megatron-LM + ModelOpt)
    - `DistillationModel` wrapper and forward hooks
    - Loss functions: KL-divergence, cosine, MSE, Top-K logits
    - Dynamic loss balancing and YAML configuration

    [:octicons-arrow-right-24: Knowledge distillation docs](megatron/knowledge_distillation/overview.md)

-   **Megatron-LM: Training Optimizations**

    ---

    Kernel-level and system-level optimizations for large-scale LLM training: parallelism strategies, communication overlap, pipeline scheduling, attention kernels, FP8 training, and Transformer Engine integration.

    - Tensor/pipeline/sequence parallelism strategies
    - Communication-compute overlap techniques
    - Flash attention variants and fusions
    - FP8 training with Transformer Engine

    [:octicons-arrow-right-24: Training optimization docs](megatron/basic_old/01-parallelism-strategies.md)

</div>
