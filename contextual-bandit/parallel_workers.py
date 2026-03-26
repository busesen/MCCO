import os
import numpy as np
import torch

from helpers import (
    build_env,
    train_adam,
)
from estimators import make_estimator_saa, make_estimator_mlmc
from utils import set_seed

# Process-local worker state
_PARALLEL_STATE = {}


def init_parallel_worker(state):
    """
    ProcessPool initializer: build heavy shared state once per worker.
    """
    torch.set_default_dtype(torch.float64)
    _PARALLEL_STATE.clear()
    _PARALLEL_STATE.update(state)
    _PARALLEL_STATE["sim_env"] = build_env(**state["env_args"])


def _build_worker_estimator(method, size_cfg):
    sim_env = _PARALLEL_STATE["sim_env"]
    clip_lambda = _PARALLEL_STATE["clip_lambda"]
    clip_thetas = _PARALLEL_STATE["clip_thetas"]

    if method == "SAA":
        n1, nu, ny = size_cfg
        est_fn = make_estimator_saa(sim_env, (nu, ny))
        lbl = f"n{n1}_{nu}_{ny}"
    else:
        n1, trunc, rate = size_cfg
        est_fn = make_estimator_mlmc(sim_env, trunc, rate, clip_lambda, clip_thetas)
        lbl = f"n{n1}_M({trunc[0]},{trunc[1]})_r({rate[0]},{rate[1]})"
    return int(n1), est_fn, lbl


def run_single_rep_worker(job):
    """
    Runs one repetition (one run_idx), saves .npy, and returns JSON metadata.
    """
    method = job["method"]
    size_cfg = job["size_cfg"]
    cfg_idx = int(job["config_idx"])
    run_idx = int(job["run_idx"])
    lr = job["lr"]
    seed_val = int(job["seed_val"])
    files_dir = job["files_dir"]
    fname = job["fname"]

    n1, est_fn, _ = _build_worker_estimator(method, size_cfg)
    set_seed(seed_val)

    x_t = torch.tensor(_PARALLEL_STATE["x_initial"], dtype=torch.get_default_dtype())
    res = train_adam(
        x_t,
        int(_PARALLEL_STATE["K"]),
        n1,
        lr,
        est_fn,
    )
    res.update({
        "config_idx": cfg_idx,
        "run_idx": run_idx,
        "step": [lr],
        "method": method,
    })
    np.save(os.path.join(files_dir, fname), res)

    return {
        "run_idx": run_idx,
        "seed_val": seed_val,
        "rec_dict": {"method": method, "file": fname, "config_idx": cfg_idx},
    }
