# SKILL: MLX on Apple Silicon

**Purpose:** Paste this document into your LLM session before asking it to write MLX code for Apple Silicon. MLX is Apple's own array framework — designed from scratch for unified memory — and LLMs have thinner training data on it than PyTorch. This skill corrects common mistakes and establishes correct idioms.

**Extends:** [`SKILL.md`](SKILL.md) — read that first.

---

## What MLX Is (and Isn't)

MLX is not a PyTorch wrapper. It is not a Metal API wrapper. It is a first-party Apple framework for numerical computing that treats unified memory as a first-class design principle rather than an afterthought.

Key architectural differences from PyTorch:

| Property | PyTorch (MPS) | MLX |
|---|---|---|
| Memory model | Discrete-GPU heritage; explicit `.to(device)` | Unified memory native; no device concept |
| Execution | Eager by default | **Lazy by default** |
| Differentiation | Autograd tape | Function transformations (`grad`, `vmap`, `vjp`) |
| Backend origin | CUDA-first, MPS added | Apple Silicon-first |
| Ecosystem | Massive | Growing; smaller but fast-moving |

---

## Rule M1: Understand Lazy Evaluation Before Writing Anything

MLX operations are **not executed when called**. They build a computation graph. The graph executes when:
- You call `mx.eval()`
- You print or inspect an array
- You convert to NumPy

**This is the most common source of confusion in LLM-generated MLX code.**

```python
import mlx.core as mx

# These lines do NOT compute anything yet
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a + b        # Graph node created, not computed
d = mx.sum(c)    # Another graph node

# THIS executes the full graph
mx.eval(d)
print(d)  # Also triggers eval

# Correct pattern: accumulate operations, eval once
# Wrong pattern: eval inside a loop on every step
```

**Why this matters for data science:** Lazy evaluation lets MLX optimize the full computation graph before executing. For complex pipelines (feature engineering → model → loss), this can be significantly faster than eager execution because the framework can fuse operations and eliminate redundant memory allocations.

---

## Rule M2: No `.to(device)` — Arrays Are Always in Unified Memory

Do not write `.to("mps")`, `.to("cpu")`, or `.cuda()`. There is no device concept in MLX. All arrays live in unified memory and are accessible to both CPU and GPU compute automatically.

```python
import mlx.core as mx

# WRONG — do not write this
x = mx.array([1.0, 2.0, 3.0]).to("gpu")  # AttributeError

# CORRECT — array is already in unified memory
x = mx.array([1.0, 2.0, 3.0])

# Specify compute device via stream if needed (advanced)
gpu_stream = mx.gpu
cpu_stream = mx.cpu
result = mx.add(a, b, stream=gpu_stream)  # Force GPU compute
mx.eval(result)
```

---

## Rule M3: Use Function Transformations, Not Autograd Tape

MLX uses JAX-style function transformations for differentiation. Do not write `loss.backward()`.

```python
import mlx.core as mx
import mlx.nn as nn

# Define model as a function / nn.Module
class LinearModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = mx.random.normal((out_features, in_features)) * 0.01
        self.bias = mx.zeros((out_features,))

    def __call__(self, x):
        return x @ self.weight.T + self.bias

# Define loss as a plain function
def mse_loss(model, X, y):
    pred = model(X)
    return mx.mean((pred - y) ** 2)

# Get gradient function via transformation
loss_and_grad = nn.value_and_grad(model, mse_loss)

# Training step
def train_step(model, optimizer, X, y):
    loss, grads = loss_and_grad(model, X, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
    return loss

# Optimizer
optimizer = optim.SGD(learning_rate=0.01)
```

---

## Rule M4: Convert to/from NumPy at Boundaries Only

MLX arrays and NumPy arrays share memory when possible (zero-copy on Apple Silicon). Convert at pipeline boundaries, not inside hot loops.

```python
import mlx.core as mx
import numpy as np

# NumPy → MLX: zero-copy when dtype matches
np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
mlx_array = mx.array(np_array)  # No copy if float32

# MLX → NumPy: triggers eval, then zero-copy
result = mx.sum(mlx_array)
mx.eval(result)
np_result = np.array(result)  # Back to NumPy

# Wrong pattern: converting inside a training loop
for batch in dataloader:
    x = mx.array(batch)   # OK if unavoidable
    # ... compute ...
    loss_np = np.array(loss)  # WRONG inside loop — triggers eval every step
    losses.append(loss_np)

# Correct: accumulate MLX arrays, convert at end
losses = []
for batch in dataloader:
    loss = compute_loss(mx.array(batch))
    losses.append(loss)
mx.eval(*losses)  # Single eval for all
losses_np = [np.array(l) for l in losses]
```

---

## Rule M5: MLX for Data Science Pipelines (Not Just ML)

MLX is useful beyond model training. For data science pipelines that currently run in NumPy, MLX can accelerate the computation-heavy steps.

```python
import mlx.core as mx
import numpy as np

# Large matrix operations (e.g., covariance estimation)
def compute_covariance_mlx(X: np.ndarray) -> np.ndarray:
    """
    Compute sample covariance matrix using MLX.
    Faster than NumPy for large X on Apple Silicon due to GPU acceleration.
    """
    X_mlx = mx.array(X, dtype=mx.float32)
    n = X_mlx.shape[0]
    X_centered = X_mlx - mx.mean(X_mlx, axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / (n - 1)
    mx.eval(cov)
    return np.array(cov)

# Batch distance computation (e.g., for clustering)
def pairwise_l2_mlx(A: mx.array, B: mx.array) -> mx.array:
    """Compute pairwise L2 distances without materializing full distance matrix."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    A_sq = mx.sum(A ** 2, axis=1, keepdims=True)
    B_sq = mx.sum(B ** 2, axis=1, keepdims=True)
    AB = A @ B.T
    dist_sq = A_sq + B_sq.T - 2 * AB
    return mx.sqrt(mx.maximum(dist_sq, 0))  # clip for numerical stability
```

---

## Rule M6: Memory-Efficient Patterns for Large Arrays

MLX lazy evaluation means large intermediate arrays may not be allocated until eval. Use this to write memory-efficient pipelines.

```python
import mlx.core as mx

def chunked_operation(data: mx.array, chunk_size: int = 10_000):
    """
    Process large arrays in chunks to avoid peak memory spikes.
    Each chunk evals independently, keeping memory flat.
    """
    n = data.shape[0]
    results = []

    for start in range(0, n, chunk_size):
        chunk = data[start:start + chunk_size]
        # Build computation graph for this chunk
        result = expensive_operation(chunk)
        # Eval this chunk — frees the graph, keeps only the result
        mx.eval(result)
        results.append(result)

    return mx.concatenate(results, axis=0)

# Check available memory before large allocations
def estimate_array_gb(shape: tuple, dtype=mx.float32) -> float:
    dtype_bytes = {mx.float32: 4, mx.float16: 2, mx.float64: 8, mx.int32: 4}
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    return (n_elements * dtype_bytes.get(dtype, 4)) / 1e9
```

---

## Quick Reference: MLX vs PyTorch MPS Decision

| Situation | Use MLX | Use PyTorch MPS |
|---|---|---|
| New project, no PyTorch dependency | ✓ | |
| Existing PyTorch codebase | | ✓ |
| Want lazy evaluation / graph optimization | ✓ | |
| Need Hugging Face / torchvision ecosystem | | ✓ |
| NumPy-style syntax preferred | ✓ | |
| JAX-style function transforms preferred | ✓ | |
| Need mature debugging tools | | ✓ (more mature) |
| On-device LLM inference | ✓ (mlx-lm) | |

---

## Installation

```bash
pip install mlx                    # Core framework
pip install mlx-lm                 # LLM inference (Llama, Mistral, etc.)
# jax-metal installs separately — see bayesian-skill.md for JAX/NumPyro
```

Requires macOS 13.5+ and Apple Silicon (M1 or later).

---

## Further Reading

- [`SKILL.md`](SKILL.md) — General Apple Silicon rules
- [`bayesian-skill.md`](bayesian-skill.md) — Bayesian modeling (uses JAX, not MLX)
- MLX documentation: [https://ml-explore.github.io/mlx/](https://ml-explore.github.io/mlx/)
- MLX examples: [https://github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)
