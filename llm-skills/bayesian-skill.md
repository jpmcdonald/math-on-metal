# SKILL: Bayesian Modeling on Apple Silicon

**Purpose:** Paste this document into your LLM session before asking it to write Bayesian or probabilistic models for Apple Silicon (M-series Macs). It corrects the most common failure modes: cardinality explosions, silent CPU fallbacks, and framework choices that ignore unified memory structure.

**Extends:** [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) — read that first for general Apple Silicon rules. This document adds Bayesian-specific patterns.

---

## The Core Problem with LLM-Generated Bayesian Code on Apple Silicon

LLMs trained on general Bayesian modeling examples will default to patterns that work on any hardware but exploit none of the Apple Silicon architecture. The specific failure modes, in order of frequency:

1. **Cartesian product materialization** — Broadcasting over all index combinations, including unobserved ones, filling unified memory with structurally zero cells
2. **Wrong framework choice** — Defaulting to PyMC CPU path when NumPyro/JAX Metal is available
3. **Silent MPS fallback** — PyTorch Bayesian code that believes it's on Metal but has silently fallen to CPU on unsupported ops
4. **Over-parallelized chains** — Running more MCMC chains than memory supports, thrashing the unified pool
5. **float64 by default** — Doubling memory footprint for negligible precision gain in most forecasting contexts

---

## Rule B1: Structure Hierarchical Models Around Observed Combinations Only

The single most impactful pattern for Bayesian work on Apple Silicon.

**The problem:** A model with three index levels — say, region (50), store (500), and SKU (2000) — has a theoretical Cartesian product of 50,000,000 cells. Most are unobserved. A naive implementation allocates that full tensor.

**Anti-pattern:**
```python
import pymc as pm
import numpy as np

# BAD: allocates full (n_regions, n_stores, n_skus) parameter tensor
# even though only ~5% of combinations are observed
with pm.Model() as model:
    # This broadcasts over ALL combinations
    mu = pm.Normal("mu", mu=0, sigma=1,
                   shape=(n_regions, n_stores, n_skus))
```

**Correct pattern — index into observed combinations only:**
```python
import pymc as pm
import numpy as np
import pandas as pd

# Assume df has columns: region_idx, store_idx, sku_idx, y
# All indices are integer-encoded, 0-based

# Only allocate parameters for observed levels
with pm.Model(coords={
    "region": region_labels,       # length n_regions
    "store": store_labels,         # length n_stores (observed only)
    "sku": sku_labels,             # length n_skus (observed only)
}) as model:

    # Global hyperpriors
    mu_global = pm.Normal("mu_global", mu=0, sigma=1)
    sigma_region = pm.HalfNormal("sigma_region", sigma=1)

    # Region-level effects — one per region (small, allocate fully)
    mu_region = pm.Normal("mu_region", mu=mu_global, sigma=sigma_region,
                          dims="region")

    # Store-level effects — index by observed store, not full grid
    sigma_store = pm.HalfNormal("sigma_store", sigma=0.5)
    mu_store = pm.Normal("mu_store",
                         mu=mu_region[df["region_idx"].values],  # ← index, don't broadcast
                         sigma=sigma_store,
                         shape=n_stores)

    # Likelihood uses integer indices into parameter vectors
    # No Cartesian product anywhere
    y_hat = mu_store[df["store_idx"].values]
    likelihood = pm.Normal("y", mu=y_hat, sigma=0.5, observed=df["y"].values)
```

**Key principle:** Parameters are sized by the number of *observed levels*, not the full index space. Lookups use integer indexing into 1D parameter vectors. The model never sees unobserved combinations.

---

## Rule B2: Use NumPyro + JAX for the Metal Path

PyMC defaults to a CPU sampler unless explicitly configured for a GPU backend. NumPyro with JAX routes MCMC through the Metal GPU via JAX's Metal plugin when available.

**Verify Metal is active before building the model:**
```python
import jax
import jax.numpy as jnp

# Check available devices
print(jax.devices())
# Should show: [METAL(id=0)] on Apple Silicon with jax-metal installed
# If it shows [CpuDevice(id=0)], install: pip install jax-metal

# Confirm Metal is the default backend
assert str(jax.default_backend()) == "metal", \
    "JAX Metal backend not active — check jax-metal installation"
```

**NumPyro model equivalent of the hierarchical pattern above:**
```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax

def hierarchical_model(region_idx, store_idx, y=None):
    n_regions = int(jnp.max(region_idx)) + 1
    n_stores = int(jnp.max(store_idx)) + 1

    # Global hyperpriors
    mu_global = numpyro.sample("mu_global", dist.Normal(0, 1))
    sigma_region = numpyro.sample("sigma_region", dist.HalfNormal(1))

    # Region effects
    with numpyro.plate("regions", n_regions):
        mu_region = numpyro.sample("mu_region",
                                   dist.Normal(mu_global, sigma_region))

    # Store effects — indexed by observed stores only
    sigma_store = numpyro.sample("sigma_store", dist.HalfNormal(0.5))
    with numpyro.plate("stores", n_stores):
        mu_store = numpyro.sample("mu_store",
                                  dist.Normal(mu_region[region_idx[:n_stores]], sigma_store))

    # Likelihood
    with numpyro.plate("observations", len(y) if y is not None else 1):
        numpyro.sample("y",
                       dist.Normal(mu_store[store_idx], 0.5),
                       obs=y)

# Run MCMC on Metal
nuts_kernel = NUTS(hierarchical_model)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=500,
    num_samples=1000,
    num_chains=4,          # See Rule B4 for chain count guidance
    progress_bar=True
)
mcmc.run(
    jax.random.PRNGKey(42),
    region_idx=region_idx_jax,
    store_idx=store_idx_jax,
    y=y_jax
)
```

---

## Rule B3: PyTorch Bayesian Models — Verify MPS and Minimize Transfers

If using PyTorch (Pyro, custom samplers, rolling forecasters), apply the device pattern consistently and verify at initialization.

**Device helper with smoke test:**
```python
import torch

def get_bayes_device(prefer_mps: bool = True) -> torch.device:
    """
    Returns MPS device if available and verified, CPU otherwise.
    Runs a smoke test using a Bayesian-relevant op (normal sample).
    """
    if prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            # Smoke test: sample from Normal — a core Bayesian op
            mu = torch.zeros(100, device=device)
            sigma = torch.ones(100, device=device)
            _ = torch.normal(mu, sigma)
            # Enable tf32 for matmul throughput (useful for covariance ops)
            torch.backends.mps.allow_tf32 = True
            return device
        except Exception as e:
            print(f"MPS smoke test failed ({e}), using CPU")
            return torch.device("cpu")
    return torch.device("cpu")
```

**Keep tensors on device through the full sampling loop:**
```python
device = get_bayes_device()

# Move all data to device once — not inside the loop
y_tensor = torch.tensor(y_values, dtype=torch.float32, device=device)
X_tensor = torch.tensor(X_values, dtype=torch.float32, device=device)

# Parameters stay on device
mu = torch.zeros(n_params, device=device, requires_grad=True)
log_sigma = torch.zeros(n_params, device=device, requires_grad=True)

# Sampling loop — no .cpu() calls inside
for step in range(n_steps):
    # All ops stay on MPS
    sigma = torch.exp(log_sigma)
    sample = mu + sigma * torch.randn_like(mu)  # randn_like respects device
    log_prob = compute_log_prob(sample, y_tensor, X_tensor)  # stays on device
    # ...

# Only move to CPU when you need numpy (e.g., for plotting, serialization)
posterior_samples = mu.detach().cpu().numpy()
```

---

## Rule B4: Chain Count and Memory Budget

MCMC chains are memory-multiplying. Each chain holds a full copy of the model's parameter state. On a machine with shared CPU/GPU memory, over-allocating chains competes with everything else.

**Memory estimation before running:**
```python
def estimate_mcmc_memory_gb(
    n_params: int,
    n_samples: int,
    n_chains: int,
    dtype_bytes: int = 4  # float32
) -> float:
    """
    Rough estimate of MCMC posterior storage.
    Actual usage higher due to warmup traces, diagnostics, etc.
    Multiply by ~2.5 for practical ceiling estimate.
    """
    raw_gb = (n_params * n_samples * n_chains * dtype_bytes) / 1e9
    return raw_gb * 2.5  # Safety multiplier

# Example: 5000 parameters, 1000 samples, 4 chains
estimate = estimate_mcmc_memory_gb(5000, 1000, 4)
print(f"Estimated memory: {estimate:.1f} GB")

# Rule of thumb: don't exceed 40% of available unified memory for MCMC
# On 64GB: ~25GB ceiling for MCMC; leave room for OS, data, model
```

**Chain count guidelines by machine:**

| Unified memory | Practical max chains | Notes |
|---|---|---|
| 16GB | 2 | Leave room for OS + data |
| 32GB | 4 | Standard configuration |
| 64GB | 4–8 | Depends on model size |
| 96GB+ | 8+ | Large hierarchical models |

**Prefer fewer chains with more samples over many chains with few samples.** Convergence diagnostics (R-hat) are more reliable with longer chains than more chains.

---

## Rule B5: Zero-Inflated and Mixture Models — Handle Sparsity Explicitly

Zero-inflated models (ZIP, ZINB) are common in demand forecasting and count data. The naive implementation broadcasts the inflation probability over all observations; the correct implementation exploits known sparsity structure.

**NumPyro ZIP with explicit zero mask:**
```python
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp

def zip_model(X, y=None):
    """
    Zero-Inflated Poisson.
    X: feature matrix (n_obs, n_features)
    y: observed counts (n_obs,) — integer
    """
    n_features = X.shape[1]

    # Coefficients for Poisson rate
    beta = numpyro.sample("beta",
                          dist.Normal(jnp.zeros(n_features),
                                     jnp.ones(n_features)))

    # Zero-inflation probability (logit parameterization for numerical stability)
    logit_pi = numpyro.sample("logit_pi", dist.Normal(0, 1))
    pi = numpyro.deterministic("pi", jax.nn.sigmoid(logit_pi))

    # Poisson rate (log-linear)
    log_mu = X @ beta
    mu = jnp.exp(jnp.clip(log_mu, -10, 10))  # clip for numerical stability

    # Likelihood: ZeroInflatedPoisson
    with numpyro.plate("obs", X.shape[0]):
        numpyro.sample("y",
                       dist.ZeroInflatedPoisson(gate=pi, rate=mu),
                       obs=y)
```

**When the zero-inflation probability varies by group (hierarchical ZIP):**
```python
def hierarchical_zip_model(X, group_idx, y=None):
    n_groups = int(jnp.max(group_idx)) + 1
    n_features = X.shape[1]

    # Shared coefficients
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(n_features),
                                              jnp.ones(n_features)))

    # Group-level zero-inflation — one logit per observed group
    mu_logit_pi = numpyro.sample("mu_logit_pi", dist.Normal(0, 1))
    sigma_logit_pi = numpyro.sample("sigma_logit_pi", dist.HalfNormal(1))

    with numpyro.plate("groups", n_groups):
        logit_pi_group = numpyro.sample("logit_pi_group",
                                        dist.Normal(mu_logit_pi, sigma_logit_pi))

    pi = jax.nn.sigmoid(logit_pi_group[group_idx])  # index, don't broadcast
    mu = jnp.exp(jnp.clip(X @ beta, -10, 10))

    with numpyro.plate("obs", X.shape[0]):
        numpyro.sample("y",
                       dist.ZeroInflatedPoisson(gate=pi, rate=mu),
                       obs=y)
```

---

## Rule B6: Rolling / Online Bayesian Updates — Checkpoint Pattern

For rolling forecasters that update posteriors as new data arrives, checkpoint state rather than re-running from scratch. On Apple Silicon, checkpointing to disk is fast (NVMe directly on the SoC) but the real saving is keeping warm-started chain state in unified memory between update cycles.

**Checkpoint pattern:**
```python
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

class RollingBayesianForecaster:
    def __init__(self, model_fn, checkpoint_dir: Path):
        self.model_fn = model_fn
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._mcmc = None
        self._last_samples = None

    def _checkpoint_path(self, period_id: str) -> Path:
        return self.checkpoint_dir / f"posterior_{period_id}.pkl"

    def fit(self, data, period_id: str, num_warmup: int = 500,
            num_samples: int = 1000, num_chains: int = 4):

        nuts = NUTS(self.model_fn)
        self._mcmc = MCMC(nuts,
                          num_warmup=num_warmup,
                          num_samples=num_samples,
                          num_chains=num_chains)

        # Warm-start from previous posterior if available
        init_params = None
        if self._last_samples is not None:
            # Use posterior mean as initialization for next period
            init_params = {k: jnp.mean(v, axis=(0, 1))
                          for k, v in self._last_samples.items()}

        self._mcmc.run(
            jax.random.PRNGKey(hash(period_id) % 2**31),
            **data,
            init_params=init_params
        )

        self._last_samples = self._mcmc.get_samples(group_by_chain=True)

        # Checkpoint to disk
        with open(self._checkpoint_path(period_id), "wb") as f:
            pickle.dump(self._last_samples, f)

        return self

    def load_checkpoint(self, period_id: str):
        path = self._checkpoint_path(period_id)
        if path.exists():
            with open(path, "rb") as f:
                self._last_samples = pickle.load(f)
        return self

    def predict(self, X_new):
        if self._last_samples is None:
            raise RuntimeError("No posterior samples — call fit() or load_checkpoint() first")
        # Posterior predictive using stored samples
        # (implementation depends on model structure)
        ...
```

---

## Convergence Diagnostics — Don't Skip These

Bayesian models on Apple Silicon run faster, which creates pressure to skip diagnostics. Don't.

**Minimum checks after every MCMC run:**
```python
import arviz as az

# Convert to ArviZ InferenceData
idata = az.from_numpyro(mcmc)

# R-hat: should be < 1.01 for all parameters
rhat = az.rhat(idata)
rhat_max = float(rhat.to_array().max())
if rhat_max > 1.01:
    print(f"WARNING: Max R-hat = {rhat_max:.3f}. Convergence not confirmed.")
    print("Consider: more warmup, more samples, or reparameterization.")

# Effective sample size: should be > 400 per chain for key parameters
ess = az.ess(idata)
ess_min = float(ess.to_array().min())
if ess_min < 400:
    print(f"WARNING: Min ESS = {ess_min:.0f}. Posterior may be unreliable.")

# Energy diagnostic: E-BFMI > 0.3 suggests no sampler pathologies
energy = az.bfmi(idata)
if any(e < 0.3 for e in energy):
    print(f"WARNING: Low E-BFMI {energy}. Possible funnel geometry — consider reparameterization.")
```

---

## Quick Reference: Framework Choice for Bayesian Work

| Situation | Recommended stack |
|---|---|
| New hierarchical model, want Metal GPU | NumPyro + JAX + jax-metal |
| Existing PyMC model, staying CPU | PyMC + CPU (it's fast enough for many models) |
| Rolling/online updates, PyTorch familiarity | PyTorch MPS + Pyro + checkpoint pattern |
| Large sparse hierarchical model | NumPyro + JAX; apply Rule B1 aggressively |
| Zero-inflated count data | NumPyro ZIP (Rule B5) or sklearn ZI wrappers on CPU |
| Needs sklearn pipeline integration | PyMC or CPU path; MPS not worth the friction here |

---

## Common Error Messages and Fixes

**`RuntimeError: MPS backend out of memory`**
→ Reduce `num_chains`, use float32 instead of float64, check for Cartesian product allocation (Rule B1)

**`UserWarning: There were X divergences after tuning`**
→ Reparameterize (non-centered parameterization for hierarchical models), increase `target_accept`, or add more warmup

**`jax.errors.UnexpectedTracerError`**
→ You have a Python conditional inside a JAX-traced function; replace with `jax.lax.cond` or restructure outside the model

**Chains converge on CPU but not on MPS/Metal**
→ Check for float precision issues; some MPS ops use lower precision than CPU; try disabling `allow_tf32`

**`R-hat > 1.1` on group-level parameters**
→ Usually a funnel geometry issue in the hierarchical prior; switch to non-centered parameterization

---

## Further Reading

- [`apple-silicon-data-science-skill.md`](apple-silicon-data-science-skill.md) — General Apple Silicon rules (read first)
- [`../../foundations/unified-memory/cpu-vs-gpu-paths.md`](../../foundations/unified-memory/cpu-vs-gpu-paths.md) — When CPU is the right answer
- [`mlx-skill.md`](mlx-skill.md) — MLX patterns for Apple Silicon
- NumPyro documentation: [https://num.pyro.ai](https://num.pyro.ai)
- JAX Metal plugin: [https://developer.apple.com/metal/jax/](https://developer.apple.com/metal/jax/)
- ArviZ diagnostics: [https://python.arviz.org](https://python.arviz.org)
