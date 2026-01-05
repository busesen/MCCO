# Distributionally Robust Contextual Bandits (MLMC vs SAA)

This folder contains the full experimental pipeline for the **distributionally robust offline contextual bandit** experiment described in the paper section:

> *Distributionally Robust Policy Learning for Offline Contextual Bandits*

The experiment compares a **Sample Average Approximation (SAA)** gradient estimator with a **truncated Multilevel Monte Carlo (MLMC)** gradient estimator for solving a **three-stage MCCO** problem.

---

## Problem Overview

We study the optimization problem

$$
\min_{\theta, \lambda \geq 0} 𝔼_{c'} \left[ \frac{1}{\mu} \log 𝔼_u \left[ \exp \left( \mu \left( 𝔼_{y|u} \left[ \sum_a \pi_\theta(a|u) y_a \right] + r_y + r_c^2 \lambda - \lambda \|u - c'\|^2 \right) \right) \right] \right]
$$

This objective arises from a **Wasserstein distributionally robust formulation** of offline contextual bandits and involves:

1. an outer expectation over contexts c',
2. a middle expectation over contexts u,
3. an inner expectation over stochastic costs y.

The goal is to estimate gradients of this objective efficiently.

---

## Dataset

**File:** `dataset.xlsx`

- Context features: `c_0` – `c_5` (6 categorical variables)
- Unique contexts: 1,440
- Synthetic population size: 9,000
- Weights:
  - `COUNT_1`: training (behavioral) distribution
  - `COUNT_2`: test distribution (shifted)

Contexts are normalized to `[0,1]` before simulation.

---

## Folder Structure

```
.
├── dataset.xlsx          # Synthetic contextual bandit dataset
├── run_ctx.py            # Main experiment runner (CLI entry point)
├── empirical_solver.py   # Exact robust solver (ground truth)
├── estimators.py         # SAA and truncated MLMC gradient estimators
├── simulators.py         # Data-generating process (contexts & costs)
├── costs.py              # Conditional cost model E[y | c]
├── helpers.py            # Training loop, projection, evaluation
├── utils.py              # Seeding and tensor utilities
├── visualization.py     # Plotting and evaluation figures
└── README.md             # This file
```

---
 * **`costs.py`**: Cost Model
     * Two actions (treatments): a0 and a1
     * Costs are stochastic and Gaussian
     * Conditional mean:
         * piecewise linear in feature `c_4`
         * split by binary indicator `c_0`
     * Test-time distribution shift applied via `shift = [s0, s1]`.


* **`empirical_solver.py`**: Exact Robust Solution

  * The file `empirical_solver.py` computes the **exact population solution** of the DRO objective using  scipy.optimize  as it is a convex optimization problem if the cost distribution is known.
  * This solution is used as a **ground-truth benchmark** for evaluating convergence of SAA and MLMC.
  * Example output:
    * Optimal parameters: (λ*, θ₁*, θ₂*) ≈ (7.96, 0.25, 0.89)

* **`estimators.py`**: Gradient Estimators

  * Sample Average Approximation (SAA): `SAA_grad()`
  * Truncated Multilevel Monte Carlo (MLMC): truncated_MLMC_grad()


* **`helpers.py`**: Optimization & Evaluation
  * Optimizer: Adam
  * Projection:
    * λ ≥ 0
    * θ₁, θ₂ ∈ [0,1]
  * Gradients optionally clipped for stability
  * After training, policies are evaluated on a **shifted test distribution**:
  
  * Covariate shift via `COUNT_2`
  * Cost shift via additive mean perturbation

Metrics:
* Exact theoretical test cost
* SAA estimate of the DRO objective
in helpers.py

* **`visualization.py`**:
  * Plots include:
  * Parameter trajectories
  * Cost vs computational budget


## 🚀 Usage

From this folder:

```bash
python run_ctx.py
```

Key command-line arguments:

- `--K` : number of SGD iterations
- `--num_runs` : number of independent runs
- `--saa_sizes` : sample sizes for SAA
- `--mlmc_sizes` : truncation levels and rates for MLMC
- `--learning_rates` : optimizer step sizes
- `--r_c`, `--r_y` : Wasserstein radii
- `--mu` : softmax temperature
