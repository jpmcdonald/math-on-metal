# CPU vs GPU Paths on Apple Silicon: What "Unified Memory" Actually Means for Your Workload

Most content about Apple Silicon for data science collapses two distinct ideas into one:

1. The CPU and GPU share a single physical memory pool
2. Therefore you should use the GPU for everything

The second does not follow from the first. This document explains the distinction, gives you a decision framework for which path to use, and is honest about where the CPU is the right answer.

---

## The Architecture, Plainly

On a discrete GPU system (any machine with a PCIe GPU), there are two memory pools:

- **System RAM** — accessible by the CPU
- **VRAM** — accessible by the GPU

Moving data between them requires a PCIe transfer. For iterative workloads — Bayesian samplers, constraint solvers, MCMC chains — this transfer cost dominates. The GPU may be faster at the compute, but if you're constantly moving data back to check convergence, adjust parameters, and re-send, the bus becomes the bottleneck.

Apple Silicon eliminates this bottleneck. There is one memory pool. The CPU and GPU both address it directly. A tensor created by the CPU is already "on the GPU" — no copy required.

**What this means:** The unified architecture removes a cost. It does not change which compute unit is appropriate for a given operation. The CPU is still better at serial work. The GPU is still better at wide parallel work. The difference is that *switching between them is now nearly free*.

---

## Three Regimes, Not Two

Most discussions treat this as a binary: CPU or GPU. On Apple Silicon the useful frame is three regimes:

### Regime 1: CPU + Threads + Unified RAM

**What it is:** Multi-core CPU execution, parallel jobs, large in-memory data structures. No GPU involvement. The "unified" part of unified memory means your data can be large — you're drawing on the full memory pool, not a separate VRAM budget.

**When it's the right choice:**
- Workloads with serial dependencies (each step depends on the previous result)
- Scikit-learn models (LogisticRegression, PoissonRegressor, GradientBoosting)
- DuckDB / SQL analytics — DuckDB uses CPU threads, not GPU
- XGBoost on CPU (GPU path requires CUDA; Apple GPU gains are marginal without it)
- Job orchestration, checkpointing, signal handling
- Large pandas / NumPy operations that don't vectorize cleanly to GPU ops

**Common mistake:** Calling this "not using the hardware." You are using the hardware. The M-series CPU is fast, the thread count is high, and the memory pool is large. CPU-threaded work on a 64GB M2 Max is legitimately powerful — just not Metal.

**Honest label:** "Exploiting the machine" ≠ "Using the GPU."

---

### Regime 2: PyTorch MPS (Metal Performance Shaders)

**What it is:** PyTorch tensor operations routed to the Apple GPU via Metal Performance Shaders. This is the primary path for Apple Silicon GPU compute in Python.

**When it's the right choice:**
- Neural network training and inference
- Large matrix operations that map naturally to GPU parallelism
- Bayesian models implemented as tensor ops (Pyro, NumPyro with PyTorch backend)
- Rolling forecasters with batched tensor operations

**The caveat that matters:** Not every PyTorch op is implemented on MPS. PyTorch silently falls back to CPU for unsupported ops. You may believe your model is running on Metal when significant portions are not.

**Always verify:**
```python
import torch

def get_device(prefer_mps: bool = True, smoke_test: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        if smoke_test:
            # Use a known-supported op to confirm the path is live
            try:
                _ = torch.zeros(1, device=device) + torch.ones(1, device=device)
            except Exception as e:
                print(f"MPS smoke test failed, falling back to CPU: {e}")
                return torch.device("cpu")
        return device
    return torch.device("cpu")
```

**Minimize CPU↔MPS copies.** Every `.to(device)` and `.cpu()` call is an opportunity to introduce unnecessary transfers. Structure your code so tensors stay on MPS for the full forward/backward pass. Profile with `torch.profiler` to confirm.

**tf32 and fp16 on MPS:**
```python
# These are often worth enabling on MPS for throughput
torch.backends.mps.allow_tf32 = True  # Faster matmuls, small precision cost
# fp16 support on MPS is version-dependent — test before enabling in production
```

---

### Regime 3: MLX (Apple-First)

**What it is:** Apple's own ML framework, designed from scratch for unified memory. Unlike PyTorch MPS, MLX doesn't carry a CUDA heritage — it's built around the assumption that CPU and GPU share memory.

**When it's the right choice:**
- New projects where you don't have an existing PyTorch dependency
- Workloads where PyTorch MPS fallbacks are causing problems
- On-device inference where you want maximum Apple Silicon efficiency
- When you want NumPy-like syntax with GPU execution

**Key difference from PyTorch MPS:** MLX uses lazy evaluation — operations are not executed until the result is needed. This allows the framework to optimize the full computation graph before executing, which can significantly outperform eager PyTorch MPS for complex operations.

```python
import mlx.core as mx

# Arrays live in unified memory — no explicit .to(device) needed
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a + b  # Lazy — not yet computed
mx.eval(c)  # Executes the graph
```

**Tradeoff:** MLX ecosystem is smaller than PyTorch. If your workflow depends on specific PyTorch integrations (Hugging Face, specific model architectures), MPS is more practical today.

---

## Decision Framework

```
Is your workload primarily:
│
├── Serial/iterative with CPU-native libraries (sklearn, DuckDB, XGBoost)?
│   └── → Regime 1: CPU + threads + unified RAM
│       (Multi-core, large in-memory data. Not a compromise — it's the right tool.)
│
├── Tensor-based, existing PyTorch codebase?
│   └── → Regime 2: PyTorch MPS
│       (Verify with smoke test. Monitor for silent CPU fallbacks. Profile.)
│
├── New project or willing to adopt a new framework?
│   └── → Regime 3: MLX
│       (Best Apple Silicon efficiency. Smaller ecosystem.)
│
└── Custom numerical kernels (FFT, custom losses, simulation loops)?
    └── CPU: Use Accelerate framework (vDSP, BLAS, LAPACK)
        GPU: Metal Shading Language via metalcompute or Swift
```

---

## What "Unified Memory" Actually Buys You in Each Regime

| Regime | What unified memory changes |
|---|---|
| CPU (Regime 1) | Larger working set. 64GB for in-memory data science is unusual hardware. No VRAM cap forcing you to chunk data for GPU. |
| PyTorch MPS (Regime 2) | No PCIe transfer cost between training steps. Iterative algorithms (MCMC, EM) are more competitive with discrete GPU than raw FLOPS suggest. |
| MLX (Regime 3) | Framework designed around the assumption — lazy evaluation + unified memory enables graph-level optimization across CPU/GPU boundaries. |
| Mixed pipelines | Moving data between CPU-based analytics (DuckDB result) and GPU-based model (PyTorch MPS tensor) is cheap. Pipeline stages don't need serialization. |

---

## The Aspirational Flag Anti-Pattern

Real production codebases — even well-engineered ones written by people who understand the hardware — frequently contain a specific gap: a `use_metal: true` or similar flag in a config file that is not actually wired to any device selection or backend routing in the code.

```yaml
# config.yaml — looks good
compute:
  n_jobs: 11
  omp_threads: 1
  use_metal: true       # ← aspirational, not functional
  timeout_per_fit: 90
```

```python
# cli.py — "Metal optimization" that is actually thread caps for CPU
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# use_metal is read from config but never drives device selection
```

The thread caps are correct and important for multi-process CPU work on Apple Silicon. But they are not Metal. The `use_metal` flag is documentation of intent, not implementation.

This pattern is worth naming because it is common, not because it is a failure. The correct response is to either wire the flag honestly or remove it. A working example of honest wiring:

```python
import torch

def resolve_torch_device(config) -> torch.device:
    """
    Honest implementation of what use_metal: true should do.
    Falls back gracefully rather than silently doing nothing.
    """
    preference = config.get("torch_device", "auto")

    if preference == "auto" or preference == "mps":
        if torch.backends.mps.is_available():
            try:
                # Smoke test
                _ = torch.zeros(1, device="mps") + torch.ones(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass

    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
```

If your codebase uses sklearn, LightGBM, or statsmodels as its primary modeling stack, those remain CPU-bound regardless of this flag. The device selection only matters for code paths that actually use PyTorch or MLX tensors. Be honest about which paths those are.

---

## The Honest Accounting

A real-world data science codebase on Apple Silicon will typically look like this:

- **Most of the pipeline:** CPU regime — SQL, pandas, feature engineering, sklearn/LightGBM models, joblib parallelism, serialization, job control
- **Specific hot paths:** MPS or MLX — model training, large tensor ops, neural inference
- **Occasional custom numerics:** Accelerate (CPU) or MSL (GPU) for things the frameworks don't cover

A per-SKU forecasting pipeline using LightGBM with `joblib.Parallel(backend='loky')` across many workers is a good example. It exploits Apple Silicon's many CPU cores and large unified memory pool effectively. The fact that it doesn't use the GPU doesn't make it a poor implementation — LightGBM on CPU with careful thread management is the right tool for that problem shape. Calling it "Math on Metal" in the sense of GPU tensor acceleration would be inaccurate. Calling it well-tuned Apple Silicon code would be correct.

The table of what's actually present in a mature production codebase:

| Component | Typical path | Metal GPU involved? |
|---|---|---|
| SQL / DuckDB analytics | CPU threads | No |
| pandas feature engineering | CPU | No |
| sklearn / LightGBM models | CPU (joblib parallel) | No |
| statsmodels, scipy | CPU | No |
| PyTorch training | MPS if wired correctly | Yes — if implemented |
| NumPyro / JAX MCMC | Metal if jax-metal installed | Yes — if implemented |
| MLX arrays | GPU (unified memory) | Yes — by design |
| Embedding retrieval (FAISS) | CPU or Metal-accelerated | Depends on build |

This is not a failure. It is an accurate map. The value of Apple Silicon for most of the top half of that table is the large unified memory pool (no VRAM ceiling forcing data chunking) and the many CPU cores. That is real and worth having. It is a different claim than "GPU-accelerated analytics."

---

## Practical Checklist Before Reaching for MPS

Before moving a workload to PyTorch MPS, confirm:

- [ ] The operation is tensor-based (not a Python loop over individual elements)
- [ ] The operation is large enough that GPU parallelism amortizes kernel launch overhead (rule of thumb: arrays > 10K elements)
- [ ] The PyTorch version you're using supports the required ops on MPS (check [PyTorch MPS coverage](https://github.com/pytorch/pytorch/issues/77764))
- [ ] You have a CPU fallback for ops that aren't MPS-supported
- [ ] You've verified with a smoke test that MPS is actually active
- [ ] You've profiled the CPU version first and confirmed it's actually the bottleneck

If you can't check all of these, start in CPU regime and migrate specific bottlenecks after profiling.

---

## Further Reading

- [`../accelerate-framework/`](../accelerate-framework/) — CPU-path Accelerate patterns (BLAS, LAPACK, vDSP)
- [`../../languages/python/`](../../languages/python/) — MLX and PyTorch MPS idioms
- [`../../llm-skills/apple-silicon-data-science-skill.md`](../../llm-skills/apple-silicon-data-science-skill.md) — Prompting LLMs to generate architecture-aware code
- [`../../llm-skills/bayesian-skill.md`](../../llm-skills/bayesian-skill.md) — Bayesian model patterns for Apple Silicon
