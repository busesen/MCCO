# Bermudan Option Pricing via Truncated MLMC

This repository contains the Python implementation for the numerical experiments described in the paper regarding **Truncated Multilevel Monte Carlo (MLMC)** estimators.

Specifically, this code solves the **Bermudan Basket Option** pricing problem. It compares the performance of the proposed Truncated MLMC estimator against the standard Untruncated MLMC estimator (Zhou et al., 2022).

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

* `estimators.py`: Core implementation of the **Truncated MLMC** and **Untruncated MLMC** estimators. Contains the recursive branching logic and cost functions.
* `option_setup.py`: Defines the specific physics of the problem (Geometric Brownian Motion simulator) and the Bermudan payoff functions.
* `run_option_parallel.py`: The main entry point. Handles argument parsing, parallel execution of replicates, confidence interval calculation, and result serialization.

## Dependencies

The code relies on standard Python scientific libraries:
```bash
pip install numpy
