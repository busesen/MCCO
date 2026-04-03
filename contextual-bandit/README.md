# Contextual Bandits

This folder contains the Python implementation for comparing **Sample Average Approximation (SAA)** gradient estimator against the (truncated) **Multilevel Monte Carlo (MLMC)** gradient estimator for solving a **three-stage MCCO** problem. 

This part focuses on reproducing the experiments in subsection "Contextual Bandits".

---

## 📄 Problem Description

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

### Methods Compared

1. **SAA:** A standard SAA gradient estimator (as in Shen et al. 2024, accounted for the ambigous costs) with sample sizes $(n_1, n_2, n_3) \in \{(100, 200, 200), (100, 300, 300), (100, 500, 500), (100, 1000, 1000)\}$.
2. **Truncated MLMC:** The proposed MLMC gradient estimatot with outer batch size $n_1 = 1000$, truncation levels $(M_1, M_2) \in \{(9, 5), (10, 6)\}$, and geometric rates $(r_1, r_2) = (0.5, 0.5)$.

---


## 🗂️ Folder Structure

```
.
├── dataset.xlsx          # Synthetic contextual bandit dataset
├── run_ctx.py            # Main experiment runner (CLI entry point)
├── empirical_solver.py   # Exact empirical solver (true optimum)
├── estimators.py         # SAA and MLMC gradient estimators
├── simulators.py         # Data-generating process (contexts & costs)
├── costs.py              # Conditional cost model E[y | c]
├── helpers.py            # Training loop and projection utilities
├── utils.py              # Seeding and tensor utilities
├── visualization.py      # Plotting utilities
└── README.md             # This file
```
---
 
* **`estimators.py`**: Contains the implementations of the two gradient estimators:

  * `SAA_grad` Sample Average Approximation (SAA) gradient estimator.
  * `truncated_MLMC_grad`: The proposed (truncated) MLMC gradient estimator.

* **`simulators.py`**: Defines the data-generating process.
  * `simulator`: Generates samples for the 3 stages (Context $c'$ $\to$ Context $u$ $\to$ Cost $y$), where the first two stages follow uniform distributions and the final stage follows a lognormal distribution.
    
* **`costs.py`**: Contains the conditional cost functions $\mathbb{E}[y|c, a]$.
     * Conditional mean:
         * piecewise linear in feature `c_5`
         * split by binary indicator `c_1`
                  
* **`empirical_solver.py`**: Computes the empirical ground truth.
  * Uses scipy.optimize to solve the convex log-sum-exp problem exactly over the finite population. This provides the optimal parameters $(\lambda^*, \theta^*)$ for benchmarking.
  * Optimal parameters are found as: (λ*, θ₁*, θ₂*) ≈ (11.829, 0.589, 0.713)


* **`helpers.py`**: Utilities for the training loop.
  * `train_adam`: ADAM training loop with a softplus parameterization for $\lambda$ and projected updates for $\theta$.
  
* **`run_ctx.py`**: The main script that runs the experiments.
  * It loads data, runs the exact solver, trains models using SAA and MLMC, and logs performance metrics.
  
* **`visualization.py`**: Handles the data visualization.
  * Generates ADAM convergence figure for $\lambda$, $\theta_1$, and $\theta_2$ versus the number of scenarios summed over 2,000 iterations.

* **`dataset.xlsx`**: Contains the empirical context population used by the simulator and solver.
  * Context features: `c_1` - `c_6` (6 categorical variables)
  * Unique contexts: 1,440
  * Population size: 9,000 
  * Weight columns: `COUNT_1`, `COUNT_2`, and `COUNT`
  * `COUNT_1` is used as the training (behavioral) distribution for sampling outer contexts
  * Context features are normalized to `[0,1]` before simulation


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

## 📚 References

* Yi Shen, Pan Xu, and Michael Zavlanos. [Wasserstein Distributionally Robust Policy Evaluation and Learning for Contextual Bandits](https://arxiv.org/abs/2309.08748). *Transactions on Machine Learning Research*. 2024. ISSN 2835-8856. Featured Certification.
