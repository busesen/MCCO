# Bermudan Option Pricing via Truncated MLMC

This repository contains the Python implementation for the numerical experiments described in the paper regarding the subsection **Pricing Bermudan Options**.

Specifically, this code solves the **Bermudan Basket Option** pricing problem. It compares the performance of the proposed Truncated MLMC estimator against the Untruncated MLMC estimator (Zhou et al., 2023).

## Problem Description

We consider a Bermudan basket option with the following parameters:
* **Assets**: $m=5$ assets following independent Geometric Brownian Motions.
* **Strike Price**: $K=100$.
* **Time Horizon**: $T=4$ exercise dates.
* **Risk-free Rate**: $\gamma=0.05$.
* **Volatility**: $\sigma=0.2$.
* **Initial State**: $\xi_1 = 100 \cdot \mathbf{1}_m$.

The problem is modeled as an optimal stopping problem where the price is the risk-neutral expected net present value of payoffs under an optimal exercise strategy.

## File Structure

* **`estimators.py`**: Contains the implementations of the two main estimators:
    * `Truncated_MLMC`: The proposed truncated MLMC estimator.
    * `Untruncated_MLMC`: The untruncated MLMC estimator.
    * Also includes cost helper functions (`cost_untruncated`, `cost_truncated`). The functions `cost_untruncated` and `cost_truncated` compute the expected cost per tree for the untruncated and truncated MLMC estimators, respectively. This cost is given by $\mathbb{E}[2^{\lambda_1}] \times \mathbb{E}[2^{\lambda_2}]$. 
* **`option_setup.py`**: Defines the specific physics of the problem (Geometric Brownian Motion simulator) and the Bermudan payoff functions.
* **`run_option_parallel.py`**: The main script that runs the experiments. It:
    * Runs parallel simulations for all estimators.
    * Saves raw results to `.npy` files and summaries to `.json` files.

## 📚 References

* Zhengqing Zhou, Guanyang Wang, Jose H Blanchet, and Peter W Glynn. [Unbiased optimal stopping via the MUSE](https://www.sciencedirect.com/science/article/pii/S0304414922002654). Stochastic Processes and their Applications, 166:104088, 2023.
