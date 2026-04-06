# Contributing to math-on-metal

Thank you for contributing. This repo exists because the Apple Silicon data science ecosystem is under-documented and under-tooled. Every tested example, corrected skill document, and benchmark result makes it more useful.

---

## What We Need Most

In rough priority order:

### 1. Tested code examples in `domains/`

Working code with reproducible results beats untested theory. If you have a real workload that runs faster on Apple Silicon than expected — or slower than it should — document it. The most valuable contributions are honest.

Include:
- The problem you're solving
- Hardware configuration (chip, unified memory size, macOS version)
- Which regime you're using (CPU/Accelerate, PyTorch MPS, MLX, Swift)
- A minimal reproducible example
- Actual measured performance (even rough timing is fine)

### 2. `llm-skills/` additions and corrections

The skill documents are prompt fragments for LLMs. If you find a pattern that consistently produces better Apple Silicon code when included in a system prompt, add it. If an existing skill produces wrong code in your experience, correct it.

Skill documents should:
- State the failure mode they fix at the top
- Show the anti-pattern explicitly (so LLMs learn to avoid it)
- Show the correct pattern with runnable code
- Be testable — someone should be able to paste it and see better results

### 3. Benchmark results

Add to `benchmarks/results/` in the format described in [`benchmarks/methodology.md`](benchmarks/methodology.md). Even simple timings are useful. We're building a picture of where Apple Silicon wins, loses, and where the wins are larger than expected.

### 4. MSL examples

Metal Shading Language is the most under-documented area. If you have working MSL kernels for data science operations — reductions, distance computations, custom loss functions, simulation loops — they belong here.

### 5. Swift data science library patterns

The Python ecosystem took decades to build. Swift data science libraries are being built now. If you're building or using Swift libraries for numerical work, document the patterns here. See [`llm-skills/swift-skill.md`](llm-skills/swift-skill.md) for context.

---

## What Doesn't Belong Here

- Proprietary or client-specific code (generalize first)
- Python code that could run identically on any hardware (no Apple Silicon specificity)
- Marketing claims without evidence
- PyTorch code that uses `.to("cuda")` as the primary path
- Anything that requires a paid Apple Developer account to use

---

## How to Contribute

1. Fork the repo
2. Create a branch: `git checkout -b add/bayesian-sparse-pattern` or `fix/msl-threadgroup-barrier`
3. Make your changes
4. Test your code on Apple Silicon hardware
5. Submit a pull request with a brief description of what you added and why

---

## Code Standards

**Python examples:**
- Python 3.10+
- Type hints on function signatures
- `float32` by default unless you explicitly need `float64` (and say why)
- A `# Tested on: M2 Max 64GB, macOS 14.x` comment on each example

**Swift examples:**
- Swift 5.9+
- Import only what you use
- `Accelerate` for numerical operations, not manual loops
- Brief inline comments explaining the Accelerate function being called

**MSL kernels:**
- Always include `#include <metal_stdlib>`
- Always include bounds check (`if (id >= n) return;`)
- Never use CUDA idioms (see [`llm-skills/msl-skill.md`](llm-skills/msl-skill.md))

**Skill documents:**
- Start with the failure mode being addressed
- Anti-pattern before correct pattern
- Runnable code examples
- `[[buffer(N)]]` indices must match Swift dispatch code

---

## Hardware Disclosure

Please note your hardware configuration in contributed examples. Results vary across M-series generations and memory configurations. We want to build a picture across:

- M1 / M1 Pro / M1 Max / M1 Ultra
- M2 / M2 Pro / M2 Max / M2 Ultra
- M3 / M3 Pro / M3 Max
- M4 / M4 Pro / M4 Max
- Memory configurations: 16GB, 32GB, 64GB, 96GB, 128GB, 192GB

If your example only works well above a certain memory threshold, say so.

---

## License of Contributions

By submitting a pull request, you agree that your contributions will be licensed under the same dual license as the rest of this repository:

- Prose and documentation → CC-BY-SA 4.0
- Code examples → Apache 2.0

Do not submit code that is copied from proprietary sources, client engagements, or repositories without compatible licenses. See [NOTICE](NOTICE) for how we handle third-party acknowledgments.

---

## Questions and Discussion

Open an issue for:
- Questions about which approach to use for a specific workload
- Performance results that don't match expectations
- Proposed additions before writing a large PR

---

## What This Repo Is Not

This is not a framework. We are not building a new library. We are documenting patterns, correcting LLM defaults, and providing benchmarks so that practitioners can make informed decisions about how to use the hardware they already have.

If you're building a Swift data science library that belongs in its own repo, link to it from here — don't merge it in. The right outcome is a growing ecosystem of libraries that practitioners can find from this repo, not a monorepo that tries to be everything.
