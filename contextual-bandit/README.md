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

## Cost Function

Defined in `costs.py`, the cost function is piecewise linear and depend on contexts `c_0` and `c_4`.

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
 
* **`estimators.py`**: Implements the gradient estimators.

  * `SAA_grad` Sample Average Approximation (SAA) estimator.
  * `truncated_MLMC_grad`: The proposed Truncated Multilevel Monte Carlo (MLMC) estimator.

* **`simulators.py`**: Defines the data-generating process.
  * `simulator`: Generates samples for the 3 stages (Context $c'$ $\to$ Context $u$ $\to$ Cost $y$).
  * `simulator_test`: Handles the test-time distribution shifts (using COUNT_2 and additive cost shifts).
    
* **`costs.py`**: Contains the known conditional cost functions $\mathbb{E}[y|c, a]$.
     * Conditional mean:
         * piecewise linear in feature `c_4`
         * split by binary indicator `c_0`
     * Test-time distribution shift applied via `shift = [s0, s1]`.
       
* **`empirical_solver.py`**: Computes the ground truth.
  * Uses scipy.optimize to solve the convex DRO problem exactly over the finite population. This provides the optimal parameters $(\lambda^*, \theta^*)$ for benchmarking.
  * Optimal parameters are found as: (λ*, θ₁*, θ₂*) ≈ (7.96, 0.25, 0.89)


* **`helpers.py`**: Utilities for the training loop.
  * `train_adam`: Generic SGD loop with ADAM as optimizer
  * `projection_x`: Enforces constraints ($\lambda \ge 0$, $\theta \in [0,1]$).
  * `evaluate_test_performance`: Compares the learned policy against the theoretical optimal cost on the test set.
  
* **`run_ctx.py`**: (Entry Point) Orchestrates the experiment.
  * It loads data, runs the exact solver, trains models using SAA and MLMC, and logs performance metrics.
  
* **`visualization.py`**: Generates plots for:
  * Convergence of $\lambda$, $\theta_1$, and $\theta_2$ over iterations.
  * * Convergence of $\lambda$, $\theta_1$, and $\theta_2$ over the number of samples generated.
  * Tail average performance vs. computational cost.


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
