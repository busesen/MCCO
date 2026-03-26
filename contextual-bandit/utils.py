import json
import numpy as np
import os
import re
import torch


def to_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        return x
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(x, dtype=dtype)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _parse(x):
    """
    Parses a string input into JSON (list/dict) if it is a string,
    otherwise returns the object as is.
    """
    return json.loads(x) if isinstance(x, str) else x


def parse_mlmc_config(raw_list):
    n1 = int(raw_list[0])
    M, b = raw_list[1], raw_list[2]
    M1, M2 = (int(M[0]), int(M[1])) if isinstance(M, list) else (int(M), int(M))
    r1, r2 = (float(b[0]), float(b[1])) if isinstance(b, list) else (float(b), float(b))
    return n1, (M1, M2), (r1, r2)


def _extract_run_timestamp(*paths):
    """
    Find run timestamp from folders named like: res_YYYYMMDD_HHMMSS
    """
    pat = re.compile(r"^res_(\d{8}_\d{6})")
    for p in paths:
        if not p:
            continue
        cur = os.path.abspath(p)
        while True:
            m = pat.match(os.path.basename(cur))
            if m:
                return m.group(1)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return "unknown"


def _normalize_lr_cfg(lr_cfg):
    """
    Accept either scalar lr or pair [lr_lambda, lr_theta].
    Returns float or tuple(float, float).
    """
    if isinstance(lr_cfg, (list, tuple)):
        if len(lr_cfg) != 2:
            raise ValueError(f"Invalid lr config {lr_cfg}. Use scalar or [lr_lambda, lr_theta].")
        return (float(lr_cfg[0]), float(lr_cfg[1]))
    return float(lr_cfg)


def _lr_to_display(lr_cfg):
    if isinstance(lr_cfg, tuple):
        return f"(lam={lr_cfg[0]:g},th={lr_cfg[1]:g})"
    return f"{lr_cfg:g}"


def _lr_to_filename_tag(lr_cfg):
    if isinstance(lr_cfg, tuple):
        return f"lam{lr_cfg[0]:g}_th{lr_cfg[1]:g}".replace(" ", "")
    return f"{lr_cfg:g}"
