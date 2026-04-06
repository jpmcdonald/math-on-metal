# SKILL: Apple Silicon Data Science Code Generation

**Purpose:** Paste this document (or the relevant sections) into your LLM session before asking it to write data science, ML, or scientific computing code for Apple Silicon (M-series Macs). It corrects the CUDA-first defaults that most models will otherwise apply.

**Audience:** Any capable LLM — Claude, GPT-4, Gemini, or local models.

---

## Hardware Context

You are generating code for Apple Silicon (M-series). The key architectural fact:

> The CPU, GPU, and Neural Engine share a single unified memory pool. There is no PCIe bus between them. Memory allocated by the CPU is directly accessible by the GPU without copying.

This changes the performance profile of several workload classes. Apply the rules below accordingly.

---

## Rule 1: Prefer the Accelerate Framework for Numerical Operations

When writing Python that calls into C/Fortran numerical routines, prefer paths that route through Apple's Accelerate framework (BLAS, LAPACK, vDSP). On a stock Python/NumPy installation built against Accelerate, this happens automatically — but verify:

```python
import numpy as np
np.show_config()  # Should show accelerate or openblas-accelerate
```

For explicit Accelerate usage in Swift or C, use the `Accelerate` module directly. Do not reach for generic BLAS bindings when the Accelerate path is available.

**MLX preference:** For Python workloads mixing CPU and GPU, prefer `mlx` over `torch` (MPS backend). MLX is designed for Apple Silicon's unified memory and avoids unnecessary copies. Example:

```python
# Prefer this on Apple Silicon:
import mlx.core as mx
x = mx.array([1.0, 2.0, 3.0])  # Lives in unified memory, accessible to CPU and GPU

# Over this (may insert unnecessary copies via MPS):
import torch
x = torch.tensor([1.0, 2.0, 3.0]).to("mps")
```

---

## Rule 2: Exploit Sparsity — Do Not Materialize Full Cartesian Products

The most common performance error on Apple Silicon Bayesian and probabilistic workloads is materializing a full Cartesian product of index variables when the model is actually sparse.

**Anti-pattern:**
```python
# BAD: Creates an N x M x K tensor in memory for all combinations
# even when most cells are structurally zero
import numpy as np
N, M, K = 1000, 500, 200
full_grid = np.zeros((N, M, K))  # 100M cells — kills unified memory budget
```

**Preferred pattern:**
```python
# GOOD: Represent only the non-zero cells with sparse index arrays
# Use COO (coordinate) format or panel data with explicit indices
import scipy.sparse as sp
# Or: represent hierarchical model structure as an index DataFrame
# with only the observed (group, subgroup, observation) triples
```

For Bayesian hierarchical models specifically: structure the model so that index variables reference only observed combinations. Pass `coords` and `dims` explicitly in PyMC or NumPyro. Do not broadcast across the full index space unless the model requires it.

---

## Rule 3: Use NumPyro with JAX (Metal backend) for Bayesian Models

PyMC on Apple Silicon defaults to a CPU-only path unless configured. NumPyro with JAX's Metal plugin routes MCMC sampling through the GPU via unified memory more efficiently.

```python
# Preferred Bayesian stack on Apple Silicon
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Verify Metal backend is active
print(jax.devices())  # Should show METAL device

# Model structure: use explicit plate notation for hierarchy
# Avoid broadcasting over unobserved index combinations
def model(group_idx, obs):
    n_groups = len(jnp.unique(group_idx))
    mu_global = numpyro.sample("mu_global", dist.Normal(0, 1))
    sigma_global = numpyro.sample("sigma_global", dist.HalfNormal(1))
    with numpyro.plate("groups", n_groups):
        mu_group = numpyro.sample("mu_group", dist.Normal(mu_global, sigma_global))
    with numpyro.plate("obs", len(obs)):
        numpyro.sample("y", dist.Normal(mu_group[group_idx], 1), obs=obs)
```

---

## Rule 4: Metal Shading Language (MSL) Patterns

MSL is C++14-based. Common mistakes when generating MSL:

- Do not use `__global__` (CUDA keyword) — use `device` address space qualifier
- Thread indexing is `thread_position_in_grid`, not `blockIdx * blockDim + threadIdx`
- `threadgroup` is the MSL equivalent of CUDA `__shared__`
- Buffer arguments use `device T* buffer [[buffer(0)]]` syntax
- Always specify `[[kernel]]` attribute for compute kernels

**Minimal correct MSL kernel:**
```metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}
```

**Python dispatch via PyObjC / ctypes:** Use `metalcompute` (pip-installable) for straightforward kernel dispatch from Python without full Objective-C/Swift bridging.

---

## Rule 5: Memory Budget Awareness

Unified memory is shared between CPU, GPU, and system. For an M2 Max 64GB machine:
- OS and system processes: ~8–12GB typical
- Available for workload: ~50GB practical ceiling
- GPU pressure from large models (e.g., local LLMs via Ollama/MLX) competes with data science workloads

When writing code that allocates large arrays:
- Check allocation size before creating: `array_size_gb = (N * M * dtype_bytes) / 1e9`
- Warn or raise if estimated size exceeds a configurable threshold (default: 8GB)
- Prefer float32 over float64 unless precision requires it — halves memory footprint with no accuracy loss for most ML workloads
- For MCMC: use `num_chains=4` and `num_warmup` tuned to model complexity; don't default to maximally parallel chains that saturate memory

---

## Rule 6: Swift + Accelerate for Performance-Critical Paths

If generating Swift code for numerical workloads, use the `Accelerate` import and its `vDSP`, `BLAS`, and `BNNS` sub-frameworks. Do not implement manual loops for operations that Accelerate covers.

```swift
import Accelerate

// Vectorized dot product
func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    var result: Float = 0
    vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
    return result
}

// FFT via vDSP
func computeFFT(_ signal: [Float]) -> [DSPComplex] {
    let log2n = vDSP_Length(log2(Float(signal.count)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
    defer { vDSP_destroy_fftsetup(fftSetup) }
    // ... (see Accelerate documentation for full FFT pattern)
}
```

---

## Rule 7: FAISS on Apple Silicon

FAISS can be built with Metal acceleration for vector similarity search. Prefer flat indices (`IndexFlatL2`, `IndexFlatIP`) for datasets that fit in unified memory — they avoid quantization error and are faster than IVF variants when memory is not the bottleneck.

```python
import faiss

# For datasets fitting in unified memory, flat index is often fastest
d = 768  # embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)  # stays in unified memory pool
D, I = index.search(query, k=10)
```

---

## Rule 8: Wire Device Selection Explicitly — No Aspirational Flags

A common pattern in production codebases: a `use_metal: true` config flag that reads well but is never checked during device selection. When generating or refactoring code, always wire device preference to actual backend selection. The canonical pattern:

```python
from dataclasses import dataclass
from typing import Literal
import torch

@dataclass
class ComputeProfile:
    """
    Explicit compute profile — replaces ad-hoc use_metal flags.
    torch_device: 'auto' | 'mps' | 'cuda' | 'cpu'
    n_jobs: number of parallel CPU workers (for joblib / sklearn)
    blas_threads: intra-op thread count per worker (set low for many-worker jobs)
    """
    torch_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    n_jobs: int = -1
    blas_threads: int = 1

    def resolve_torch_device(self) -> torch.device:
        if self.torch_device in ("auto", "mps"):
            if torch.backends.mps.is_available():
                try:
                    _ = torch.zeros(1, device="mps") + torch.ones(1, device="mps")
                    return torch.device("mps")
                except Exception:
                    pass
        if self.torch_device in ("auto", "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def apply_blas_env(self):
        """
        Cap BLAS threads to prevent oversubscription in multi-worker jobs.
        Critical for Apple Silicon CPU-parallel workloads.
        """
        import os
        t = str(self.blas_threads)
        os.environ["OMP_NUM_THREADS"] = t
        os.environ["MKL_NUM_THREADS"] = t
        os.environ["OPENBLAS_NUM_THREADS"] = t
        os.environ["VECLIB_MAXIMUM_THREADS"] = t  # Apple-specific

# Usage
profile = ComputeProfile(torch_device="auto", n_jobs=11, blas_threads=1)
profile.apply_blas_env()
device = profile.resolve_torch_device()

# Note: sklearn, LightGBM, statsmodels, DuckDB remain CPU-bound
# regardless of this profile. Device selection only affects PyTorch/MLX paths.
```

**The key discipline:** when you write `use_metal: true` in a config, always ask — where in the code does this actually select a device? If the answer is nowhere, either wire it or remove it.

| Anti-pattern | Why it's wrong on Apple Silicon | Preferred alternative |
|---|---|---|
| `.to("cuda")` | No CUDA on Apple Silicon | `.to("mps")` or use MLX natively |
| `torch.multiprocessing` with GPU tensors | MPS doesn't support fork-based multiprocessing | Use `spawn` context or JAX's vmap |
| Full Cartesian product allocation | Fills unified memory; no separate GPU VRAM to overflow to | Sparse indices, explicit plate notation |
| float64 by default | 2x memory for negligible accuracy gain in most ML workloads | float32 unless precision required |
| CUDA-specific kernel syntax in MSL | Compiler error | Use MSL address space qualifiers |
| Ignoring Accelerate for basic BLAS | Leaves 2–4x performance on the table | Route through Accelerate/MLX |

---

## Quick Decision Guide

**"Should I use MLX, PyTorch MPS, or CPU NumPy?"**

- Pure tensor/array math, model inference → **MLX**
- Existing PyTorch codebase with MPS support → **PyTorch MPS**
- Symbolic computation, constraint solving, MCMC with NumPy-backed samplers → **CPU NumPy + Accelerate**
- Need Metal compute kernels directly → **MSL via metalcompute or Swift**

**"My Bayesian model is slow — where do I start?"**
1. Check if you're materializing a full index grid (Rule 2)
2. Check JAX Metal device is active (Rule 3)
3. Check memory allocation size (Rule 5)
4. Check MCMC chain count vs available memory

---

## Domain-Specific Skill Extensions

For more detailed guidance on specific workloads, see:

- [`accelerate-skill.md`](accelerate-skill.md) — BLAS, LAPACK, vDSP patterns
- [`mlx-skill.md`](mlx-skill.md) — MLX arrays, JIT compilation, custom ops
- [`bayesian-skill.md`](bayesian-skill.md) — Hierarchical models, sparse indices, MCMC tuning
- [`msl-skill.md`](msl-skill.md) — Metal Shading Language, kernel dispatch, thread geometry
