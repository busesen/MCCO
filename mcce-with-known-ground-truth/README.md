# MCCE with a Known Ground-Truth Solution 

This repository contains the Python implementation for comparing **Truncated Multilevel Monte Carlo (MLMC)** estimators against **Sample Average Approximation (SAA)** and **Untruncated MLMC** for Multistage Conditional Compositional Estimation (MCCE).

This part focuses on reproducing the experiments in subsection “MCCE with a Known Ground-Truth Solution”. 

## 📄 Experiment Description

We consider a synthetic instance of the MCCE problem with $T=3$ stages. The nested expectation problem is defined by the following integrands:

$$f_1(\xi_1, x_1) = \sin(\xi_1 + x_1)$$
$$f_2(\xi_2, x_2) = \sin(\xi_2 - x_2)$$
$$f_3(\xi_3, x) = \xi_3$$

**Disturbances:**
The disturbances follow a Markovian structure:
1. $\xi_1 \sim \mathcal{N}(\pi/2, 1)$
2. $\xi_2 \mid \xi_1 \sim \mathcal{N}(\xi_1, 1)$
3. $\xi_3 \mid \xi_2 \sim \mathcal{N}(\xi_2, 1)$

**Ground Truth:**
This problem possesses a closed-form solution:
$$F(x) = \exp(-1/2) \approx 0.6065$$

### Methods Compared

1.  **SAA1:** A standard SAA with uniform branching factors $n_1 = n_2 = n_3$.
2.  **SAA2:** An SAA estimator emphasizing the first stage: $n_1 = n_2^2 = n_3^2$.
3.  **Untruncated MLMC:** Uses geometric branching with rates $r_1=0.74, r_2=0.60$. The rates are adapted from Syed and Wang, ICML, 2023. 
4.  **Truncated MLMC:** Uses geometric branching with rates $r_1 \approx 0.65, r_2 \approx 0.58$ and truncation levels $M_1=6, M_2=5$.

* **`estimators.py`**: Contains the implementations of the three main estimators:
    * `Truncated_MLMC`: The proposed truncated MLMC estimator.
    * `Untruncated_MLMC`: The untruncated MLMC estimator.
    * `SAA`: The Sample Average Approximation estimator.
    * Also includes cost helper functions (`cost_untruncated`, `cost_truncated`, `cost_saa`). The functions `cost_untruncated` and `cost_truncated` compute the expected cost per tree for the untruncated and truncated MLMC estimators, respectively. This cost is given by $\mathbb{E}[2^{\lambda_1}] \times \mathbb{E}[2^{\lambda_2}]$. The function `cost_saa` computes the deterministic cost per tree, which equals $n_2 \times n_3$.

* **`problem_setup.py`**: Defines the MCCE problem instance used in the experiments.
    * `simulator_gaussian_martingale`: The simulator for the disturbances $\xi_t$.
    * `funs`: The list of functions $f_1, f_2, f_3$.
    * `true_val`: The ground truth value ($e^{-0.5}$) used for calculating mean squared error.

* **`run_test.py`**: The main script that runs the experiments. It:
    * Runs parallel simulations for all estimators.
    * Saves raw results to `.npy` files and summaries to `.json` files.

* **`visualization.py`**: Handles all data visualization.
    * Generates the Log-MSE vs. Log-Cost plot.
    * Generates the Running Average plots with 95% Confidence Intervals.

## 🚀 Usage
First, ensure you have the required dependencies installed (see `requirements.txt`).
To run the full experiment with the default settings (10 replicates, horizon $T=3$, default rates), simply run:

```bash
python run_test.py
```

## 📚 References

* Yasa Syed and Guanyang Wang. [Optimal randomized multilevel Monte Carlo for repeatedly nested expectations](https://proceedings.mlr.press/v202/syed23a.html). In International Conference on Machine Learning, pages 33343–33364, 2023.
