# Multistage Conditional Compositional Optimization — Numerical Experiments

This repository contains the Python implementations used to generate the numerical experiments in the paper  
**_Multistage Conditional Compositional Optimization_**.

The experiments evaluate **Truncated Multilevel Monte Carlo (MLMC)** methods for multistage conditional compositional optimization (MCCO) problems and compare them against relevant baselines across two representative settings:

1. A synthetic MCCE problem with a **known ground-truth solution**, enabling precise bias–variance–cost comparisons.
2. A **Bermudan basket option pricing** problem, illustrating performance in a realistic optimal stopping setting.

Each experiment is self-contained and organized in a separate subfolder.

---

## Repository Structure

```
.
├── mcce-with-known-ground-truth/
├── option-pricing/
└── README.md
└── requirements.txt
```

### `mcce-with-known-ground-truth/`

This folder reproduces the experiments from the subsection  
**“MCCE with a Known Ground-Truth Solution”**.

* Studies a synthetic three-stage MCCE problem with Markovian Gaussian disturbances.
* The nested expectation admits a closed-form solution, allowing direct evaluation of mean squared error.
* Compares:
  - Sample Average Approximation (SAA)
  - Untruncated MLMC (Syed and Wang, 2023)
  - Truncated MLMC
* Includes implementations of estimators, cost models, experiment scripts, and visualization utilities.

See `mcce-with-known-ground-truth/README.md` for full details.

---

### `option-pricing/`

This folder reproduces the experiments from the subsection  
**“Pricing Bermudan Options”**.

* Considers a Bermudan basket option pricing problem under a Geometric Brownian Motion model.
* Formulated as a multistage optimal stopping problem.
* Compares:
  - Untruncated MLMC (Zhou et al., 2023)
  - Truncated MLMC (proposed)
* Focuses on estimator accuracy versus computational cost in a financial application.

See `option-pricing/README.md` for full details.

---

## Requirements and Usage

Each subfolder contains its own experiment scripts and relies on standard scientific Python packages listed in the corresponding `requirements.txt`. We use Python 3.12.3 for our experiments. Instructions on how to run the code for each experiment can be found in the corresponding subfolders.

---

## References

* Yasa Syed and Guanyang Wang. [Optimal randomized multilevel Monte Carlo for repeatedly nested expectations](https://proceedings.mlr.press/v202/syed23a.html). In *International Conference on Machine Learning*, pages 33343–33364, 2023.

* Zhengqing Zhou, Guanyang Wang, Jose H Blanchet, and Peter W Glynn. [Unbiased optimal stopping via the MUSE](https://www.sciencedirect.com/science/article/pii/S0304414922002654). *Stochastic Processes and their Applications*, 166:104088, 2023.
