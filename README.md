# math-on-metal

> Prompts, patterns, and libraries for data science and ML on Apple Silicon — leveraging unified memory architecture for workloads that don't fit the GPU-native mold.

---

## The Thesis

Apple Silicon's unified memory architecture is underappreciated for a specific class of workloads: those with serial dependency structure, high-cardinality state, or tight CPU-GPU memory coupling. Most data science tooling is designed around the PCIe-bus assumption — discrete CPU and GPU memory pools, latency-tolerant transfer, wide parallelism.

That assumption is wrong on Apple Silicon. The CPU, GPU, and Neural Engine share the same physical memory. There is no transfer bottleneck. For workloads where that bottleneck was the binding constraint, the performance profile changes qualitatively.

This repo documents what that means in practice:

- Which workloads benefit most (and why)
- Which compiled libraries exploit the architecture correctly
- How to prompt LLMs to generate code that takes advantage of it
- Reproducible benchmarks against CUDA baselines

---

## Who This Is For

- Data scientists running serious workloads on M-series Macs who are tired of being told to "just use a cloud GPU"
- ML engineers exploring on-device inference and reasoning pipelines
- Researchers in Bayesian modeling, constraint solving, and symbolic computation
- Anyone building LLM-assisted data science tooling and wanting higher-quality generated code

---

## Repository Structure

Layout below matches the repository as it exists today. Directories without prose or examples yet contain a `.gitkeep` placeholder so Git tracks the folder. Following usual GitHub practice for a single-purpose docs repo, subdirectory READMEs are not duplicated here—the tree below is the canonical map.

```
math-on-metal/
│
├── README.md                    ← You are here
├── CONTRIBUTING.md
├── LICENSE                      # Apache 2.0 — code examples
├── LICENSE-DOCS                 # CC-BY-SA 4.0 — documentation / prose
├── NOTICE                       # Third-party acknowledgments
│
├── foundations/
│   ├── unified-memory/          # Architecture primer, why it matters
│   │   └── cpu-vs-gpu-paths.md
│   ├── accelerate-framework/    # BLAS, LAPACK, vDSP patterns (stub)
│   └── memory-management/       # Cardinality, allocation strategies (stub)
│
├── languages/
│   ├── python/                  # NumPy, JAX, MLX idioms (stub)
│   ├── swift/                   # Swift + Accelerate patterns (stub)
│   ├── c-cpp/                   # Low-level Metal interop (stub)
│   └── msl/                     # Metal Shading Language examples (stub)
│
├── domains/
│   ├── bayesian/                # Hierarchical models, MCMC, sparse handling (stub)
│   ├── optimization/            # Constraint solving on Apple Silicon (stub)
│   ├── time-series/             # Forecasting, temporal models (stub)
│   └── embeddings/              # FAISS, vector retrieval (stub)
│
├── llm-skills/                  # Prompt engineering for Apple Silicon code generation
│   ├── apple-silicon-data-science-skill.md   # Base skill — start here (paste into LLM context)
│   ├── accelerate-skill.md      # Accelerate / BLAS / LAPACK / vDSP prompts
│   ├── mlx-skill.md
│   ├── bayesian-skill.md
│   ├── msl-skill.md
│   └── swift-skill.md           # Swift + Metal / Accelerate patterns
│
└── benchmarks/
    ├── methodology.md
    └── results/                 # Benchmark outputs (stub)
```

**Populated today:** `foundations/unified-memory/cpu-vs-gpu-paths.md`, `benchmarks/methodology.md`, and the files under `llm-skills/`. **Stubs:** empty directories (and `benchmarks/results/`) hold `.gitkeep` until guides, examples, or benchmark artifacts are added.

---

## Two Levels of "Math on Metal"

This repo addresses two distinct performance layers, and it's worth being explicit about both:

**Layer 1 — Python on Apple Silicon**
Using the right Python libraries (MLX, JAX/Metal, NumPy via Accelerate, PyTorch MPS) to exploit unified memory from within the Python ecosystem. This is the lower barrier, higher compatibility path. The `llm-skills/` documents help LLMs generate code that actually uses the hardware.

**Layer 2 — Swift + Metal (the ceiling)**
Python has a fundamental constraint: the Global Interpreter Lock (GIL) means Python cannot natively use multiple CPU cores in parallel. On an M-series chip with dozens of CPU cores, a Python process runs like a single-cylinder engine in a V12. Swift is a compiled language that addresses the hardware directly — all CPU cores, the GPU via Metal, and the Neural Engine via CoreML. For the right workloads, the performance differential is not incremental. It changes what's economically feasible.

[Errol Brandt](https://www.linkedin.com/in/errolbrandt/) is actively building Swift-native data science libraries that prove this out in practice — including an Apple-native port of Pandas backed by low-level C libraries linked with Metal GPU acceleration. His benchmarks show Swift + Metal running more than 80x faster than equivalent Python for certain data-intensive workflows. The library ecosystem is still being built, but the results are worth paying attention to.

This repo documents what exists, what patterns work, and how to prompt LLMs to generate correct Swift + Metal code. See [`llm-skills/accelerate-skill.md`](llm-skills/accelerate-skill.md), [`llm-skills/swift-skill.md`](llm-skills/swift-skill.md), and [`llm-skills/msl-skill.md`](llm-skills/msl-skill.md).

---

## Where to Start

**New to Apple Silicon for data science?** Start with [`foundations/unified-memory/`](foundations/unified-memory/) — it explains why the architecture matters and which workloads benefit.

**Using LLMs to write your code?** Start with [`llm-skills/apple-silicon-data-science-skill.md`](llm-skills/apple-silicon-data-science-skill.md) — it's a prompt document you can paste directly into Claude, ChatGPT, or any capable model to get Apple Silicon-aware code generation.

**Working on a specific domain?** Jump to [`domains/`](domains/) for patterns relevant to your problem type.

---

## The `llm-skills/` Directory

This is the differentiator. Most repos stop at code examples. This one also documents *how to prompt LLMs to generate correct Apple Silicon code*.

LLMs trained on general Python/NumPy/PyTorch code will default to patterns optimized for CUDA and discrete GPU architectures. Left unprompted, they will:

- Use PyTorch backends that don't exploit the unified memory bus
- Generate Bayesian models that materialize full Cartesian products instead of exploiting sparsity
- Ignore the Accelerate framework entirely
- Write Metal Shading Language incorrectly due to sparse training data

The skill documents in `llm-skills/` are structured prompt fragments. Paste them into your session context before asking an LLM to write data science code, and the generated output will be architecture-aware.

---

## Workloads That Benefit Most

| Workload type | Why unified memory helps |
|---|---|
| Bayesian hierarchical models | High-cardinality state fits in unified pool; no GPU↔CPU transfer during sampling |
| Constraint solving (MIP, SAT) | Serial dependency structure; memory latency dominates over parallelism |
| Large embedding retrieval | FAISS flat indices can stay on-device without paging |
| Symbolic computation | Z3, clingo operate on compact symbolic state, not wide matrix ops |
| Mixed CPU/GPU pipelines | Eliminates transfer bottleneck between stages |

---

## Workloads That Don't Benefit (or Belong Elsewhere)

Be honest: if your workload is pure matrix multiplication at scale (large transformer training, dense convnets), a cloud A100 will outperform M-series hardware. This repo is not for that use case.

It is for workloads where:
- Memory bandwidth to the compute unit is the bottleneck, not raw FLOPS
- Serial dependencies limit parallelization
- The problem fits in 24–192GB of unified memory
- Latency matters more than throughput

---

## Hardware Assumptions

Most examples are developed and tested on:
- Apple M2 Max, 64GB unified memory
- Apple M3 Max / M4 Max configurations are expected to work with minor tuning
- M1 chips should work for most examples; memory-intensive examples may require adjustment

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The highest-value contributions right now:

1. Tested code examples in `domains/` with reproducible benchmarks
2. Additional `llm-skills/` prompt documents for specific domains or libraries
3. Benchmark results from different M-series configurations
4. MSL examples — this is the most under-documented area

---

## License

This repository uses a dual license reflecting its mixed content:

- **Documentation and prose** (README, foundations/, llm-skills/ skill documents, CONTRIBUTING): [Creative Commons Attribution-ShareAlike 4.0 International](LICENSE-DOCS) — free to use, share, and adapt with attribution; derivatives must carry the same license.
- **Code examples** (Python, Swift, Metal Shading Language, Shell): [Apache License 2.0](LICENSE) — free to use including commercially, with attribution; no share-alike requirement on your own code.

If you're unsure which applies: prose → CC-BY-SA 4.0. Code snippets → Apache 2.0. See [NOTICE](NOTICE) for third-party acknowledgments.

---

## An Honest Note on "Math on Metal" in Practice

Real production codebases — even those written by people who care about this — often end up mostly in CPU regime. A forecast pipeline using LightGBM and joblib across dozens of SKUs is exploiting Apple Silicon's many cores and large unified memory pool. That is legitimate and valuable. It is not the same as routing tensor work through Metal GPU.

The gap between aspiration and implementation shows up in a specific anti-pattern: a `use_metal: true` flag in a config file that isn't actually wired to anything in the code. The flag feels right to write. But unless it drives device selection in the training path, it's documentation, not engineering.

This repo tries to be specific about both: what CPU-regime work on Apple Silicon actually looks like, and what the additional steps are to genuinely route work through MPS or Metal. See [`foundations/unified-memory/cpu-vs-gpu-paths.md`](foundations/unified-memory/cpu-vs-gpu-paths.md) for the full decision framework.

---

## Acknowledgments

Built by practitioners who got tired of being told Apple Silicon "isn't serious hardware" for data science — and equally tired of Apple Silicon being oversold without honest benchmarks. The hardware is genuinely good. Use it correctly, and be honest about which parts of "correctly" you've actually implemented.

Hat tip to [Errol Brandt](https://www.linkedin.com/in/errolbrandt/) for pushing the Swift + Metal angle further than anyone else in the data science community right now.
