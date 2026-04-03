# helpers.py
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Callable, Union, Sequence
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_env(
    data_path,
    r_c,
    r_y,
    mu,
    covariance,
    cost_params,
    shift,
    gamma_1=0.0,
    gamma_2=0.0,
    dtype=None,
):
    """
    Initializes the simulation environment by loading data, normalizing features, 
    and preparing tensors for the simulator.

    Data Columns & Weights:
    - 'c_1'...'c_6': Context features.
    - 'COUNT_1': Represents weights of patients in the TRAINING set (used for optimization/solver).

    Args:
        data_path (str): Path to the .xlsx dataset file containing context columns
                         and 'COUNT_1' weights.
        r_c (float): Context distribution shift radius (Wasserstein).
        r_y (float): Cost distribution shift radius (Wasserstein).
        mu (float): Softmax (inverse) temperature:  exp (mu * ...)
        covariance (list or array): 2x2 covariance matrix for the Gaussian noise added to costs.
        cost_params (list or array): Parameters defining the synthetic linear cost functions.
        shift (list or array): Additive shift [s0, s1] for the cost model.
        dtype (torch.dtype, optional): PyTorch data type for all tensors (default: torch.get_default_dtype()).

    Returns:
        Dict[str, Any]: A dictionary `SIM_ENV` containing:
            - "data_tensor": The normalized context features (N=1440, 6).
            - "weights_prob": Normalized probability weights for the outer distribution P_{c'}.
            - "chol_cov": Pre-computed Cholesky factor of the covariance matrix.
            - Tensors for all parameters (r_c, r_y, mu, cost_params, etc.).
            - Original pandas DataFrame ("data").
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)

    data = pd.read_excel(data_path)
    CTX_COLS = ["c_1", "c_2", "c_3", "c_4", "c_5", "c_6"]

    # Scale contexts to [0, 1] (per-column)
    for c in CTX_COLS:
        m = data[c].abs().max()
        if m > 0:
            data[c] = data[c] / m

    data_tensor = torch.tensor(data[CTX_COLS].values, dtype=dtype)

    # Weights for outer sampling (Time 3)
    weights_np = data.get("COUNT_1", pd.Series(np.ones(len(data)))).values
    weights = torch.tensor(weights_np, dtype=dtype)
    weights_prob = weights / weights.sum()

    cp_t = torch.tensor(cost_params, dtype=dtype) if cost_params is not None else None
    cv_t = torch.tensor(covariance, dtype=dtype)
    chol_cov = torch.linalg.cholesky(cv_t)

    SIM_ENV = {
        "data_tensor": data_tensor,
        "weights_prob": weights_prob,
        "CTX_COLS": CTX_COLS,
        "r_y": torch.tensor(r_y, dtype=dtype),
        "r_c": torch.tensor(r_c, dtype=dtype),
        "mu": torch.tensor(mu, dtype=dtype),
        "covariance": cv_t,
        "chol_cov": chol_cov,
        "cost_params": cp_t,
        "shift": shift,
        "gamma_1": torch.tensor(gamma_1, dtype=dtype),
        "gamma_2": torch.tensor(gamma_2, dtype=dtype),
        "data": data,
    }
    return SIM_ENV
# ──────────────────────────────────────────────────────────────────────────────
#  OPTIMIZER (ADAM)
# ──────────────────────────────────────────────────────────────────────────────

Estimator = Callable[[torch.Tensor], Tuple[torch.Tensor, float]]

def train_adam(
    x_initial: torch.Tensor,
    K: int,
    n1: int,
    lr: Union[float, Sequence[float]],
    estimator: Estimator,
    eps: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Adam loop with mixed constraints:
    - lambda = softplus(lambda_raw) (no hard projection for lambda)
    - theta is optimized directly and projected to (eps, 1-eps) after each step
    """
    def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
        # Stable inverse: y + log(-expm1(-y))
        return y + torch.log(-torch.expm1(-y))

    # Support either:
    # - scalar lr: same step size for all components
    # - pair [lr_lambda, lr_theta]: separate step sizes for lambda vs (theta1, theta2)
    if isinstance(lr, (list, tuple, np.ndarray)):
        if len(lr) != 2:
            raise ValueError("If 'lr' is a sequence, it must have exactly 2 values: [lr_lambda, lr_theta].")
        lr_lambda = float(lr[0])
        lr_theta = float(lr[1])
    else:
        lr_lambda = float(lr)
        lr_theta = float(lr)

    x0 = x_initial.detach().clone()
    lam0 = torch.clamp(x0[0:1], min=float(eps))
    th = torch.clamp(x0[1:], min=float(eps), max=1.0 - float(eps)).detach().clone().requires_grad_(True)
    lam_raw = _inverse_softplus(lam0).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": [lam_raw], "lr": lr_lambda},
            {"params": [th], "lr": lr_theta},
        ],
        amsgrad=False,
    )

    lam = torch.nn.functional.softplus(lam_raw)
    xk = torch.cat([lam, th], dim=0)
    x_traj = [xk.detach().clone()]
    cost_traj = [0.0]
    iter_cost = 0.0

    for _ in tqdm(range(K)):
        optimizer.zero_grad()

        lam = torch.nn.functional.softplus(lam_raw)
        xk = torch.cat([lam, th], dim=0)

        Gbar, c = estimator(xk, batch_size=n1)
        iter_cost += c
        cost_traj.append(iter_cost)

        # Chain rule only for lambda raw coordinate:
        # dF/dlambda_raw = dF/dlambda * sigmoid(lambda_raw)
        dlam_draw = torch.sigmoid(lam_raw)
        lam_raw.grad = (Gbar[0:1] * dlam_draw).detach().clone()
        th.grad = Gbar[1:].detach().clone()

        optimizer.step()

        # Project theta only
        with torch.no_grad():
            th.clamp_(min=float(eps), max=1.0 - float(eps))

        lam = torch.nn.functional.softplus(lam_raw)
        xk = torch.cat([lam, th], dim=0)
        x_traj.append(xk.detach().clone())

    X = torch.stack(x_traj, dim=0).cpu().numpy()

    return {
        "lambda_": X[:, 0], "theta1": X[:, 1], "theta2": X[:, 2],
        "cost_history": np.array(cost_traj),
    }

def cost_truncated(truncation, rate, T):
    product = 1
    for i in range(T-1):
        r = rate[i]
        M = truncation[i]
        if r == 0.5:
            cost = (M + 1) * r / (1 - (1 - r) ** (M + 1))
        else:
            cost = r / (1 - (1 - r) ** (M + 1)) * (1 - (2 - 2 * r) ** (M + 1)) / (2 * r - 1)
        product *= cost   
    return product  
