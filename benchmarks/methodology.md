# Benchmark Methodology

Benchmarks in this repo follow a consistent methodology so results are comparable across contributors and hardware configurations.

---

## Hardware Disclosure (Required)

Every result must include:

```
Chip: Apple M2 Max (example)
Unified memory: 64GB
macOS: 14.4.1
Python: 3.11.x
Relevant library versions: mlx 0.16.x / torch 2.3.x / numpy 1.26.x
```

---

## Timing Methodology

**Python benchmarks:** Use `time.perf_counter()` for wall-clock timing. Warm up the function with one un-timed call before measuring. Report median of 5 runs, not mean (outliers from OS scheduling skew mean).

```python
import time
import statistics

def benchmark(fn, *args, n_runs: int = 5, warmup: int = 1):
    # Warmup
    for _ in range(warmup):
        fn(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "runs": n_runs
    }
```

**For MLX:** Call `mx.eval()` inside the timed region. MLX is lazy — timing without eval measures graph construction, not computation.

```python
def benchmark_mlx(fn, *args, n_runs: int = 5):
    import mlx.core as mx

    # Warmup + eval
    result = fn(*args)
    mx.eval(result)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        mx.eval(result)
        end = time.perf_counter()
        times.append(end - start)

    return {"median_s": statistics.median(times), "min_s": min(times)}
```

**For PyTorch MPS:** Synchronize before stopping the timer. MPS operations are asynchronous.

```python
def benchmark_mps(fn, *args, n_runs: int = 5):
    import torch

    # Warmup
    result = fn(*args)
    torch.mps.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        torch.mps.synchronize()  # Wait for GPU to finish
        end = time.perf_counter()
        times.append(end - start)

    return {"median_s": statistics.median(times), "min_s": min(times)}
```

---

## What to Compare Against

Wherever possible, provide a comparison baseline:

1. **Python + NumPy (CPU)** — the default most practitioners use
2. **Python + MLX or PyTorch MPS** — the Apple Silicon GPU path
3. **Swift + Accelerate** — the compiled CPU path (where applicable)
4. **Swift + Metal** — the compiled GPU path (where applicable)

Not every benchmark needs all four. But at minimum: the "naive Python" baseline and the best Apple Silicon path you've found.

---

## Result Format

Results go in `benchmarks/results/` as markdown files. Use this template:

```markdown
# [Operation Name] Benchmark

**Date:** YYYY-MM-DD
**Hardware:** Apple M_ [Max/Pro/Ultra], _GB unified memory
**macOS:** _._._
**Python:** 3._.x

## Operation

Brief description of what's being benchmarked and why it's representative.

## Results

| Implementation | Median time | vs NumPy CPU |
|---|---|---|
| NumPy CPU (baseline) | Xs | 1.0x |
| PyTorch MPS | Xs | Nx faster |
| MLX | Xs | Nx faster |
| Swift + Accelerate | Xs | Nx faster |

## Code

[Link to example in domains/ or inline snippet]

## Notes

Anything notable: memory usage, which operations are hot, surprising results, known caveats.
```

---

## Honesty Standards

**Report actual results, including disappointing ones.** A benchmark showing that PyTorch MPS is slower than NumPy CPU for a given operation is more useful than no benchmark. We want an honest map of where Apple Silicon wins and where it doesn't.

**Note silent fallbacks.** If you suspect PyTorch MPS is falling back to CPU for certain ops, note it. Use the profiler to verify. A "fast" MPS result that's actually running on CPU is a misleading benchmark.

**Don't cherry-pick data sizes.** If Apple Silicon wins at 1M elements but loses at 10K elements (common — GPU kernel launch overhead dominates small arrays), report both.
