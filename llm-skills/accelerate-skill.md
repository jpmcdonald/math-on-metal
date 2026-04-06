# SKILL: Apple Accelerate (BLAS, LAPACK, vDSP)

**Purpose:** Paste this document into your LLM session before asking it to write numerical code that should route through Apple's Accelerate framework — from Swift, C/C++, or Python stacks that link Accelerate-backed BLAS/LAPACK.

**Extends:** [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) — read that first for unified-memory context and stack choices.

---

## When to Use This Skill

Use when the task involves:

- Dense linear algebra (BLAS/LAPACK) on Apple Silicon
- Digital signal processing (vDSP, vImage-related numerical paths)
- Verifying that Python NumPy/SciPy is actually using Accelerate rather than a generic OpenBLAS build

---

## Rule A1: Prefer Accelerate-Native Entry Points

In **Swift**, import and use `Accelerate` directly for vDSP and LAPACK wrappers rather than ad-hoc C BLAS bindings when the API surface exists.

In **Python**, prefer builds where `numpy.show_config()` reflects an Accelerate-linked BLAS. If the environment uses a different BLAS, call out the performance and correctness implications for Apple Silicon.

In **C/C++**, link against Apple's Accelerate-provided symbols where documented rather than assuming a Linux-centric BLAS layout.

---

## Rule A2: Do Not Assume CUDA-Style Memory Models

Accelerate operates in the CPU unified-memory pool (with vectorization and AMX where applicable). Generated code should not insert unnecessary copies “for the GPU” when the requested operation is a CPU Accelerate call.

---

## Content Roadmap

Concrete examples for `foundations/accelerate-framework/` and language-specific notes in `languages/swift/` and `languages/python/` will land in this repository over time. Until then, combine this skill with [`swift-skill.md`](swift-skill.md) for Swift-heavy tasks and [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) for Python MLX/JAX/NumPy defaults.
