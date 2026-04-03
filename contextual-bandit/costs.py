import numpy as np
import torch

def cost_ymean(u, params, shift=None, Test=False):
    # ---------------------------------------------------------
    # Helper: Force everything to be a Tensor
    # ---------------------------------------------------------
    def to_t(x):
        if torch.is_tensor(x):
            return x
        # Safely handle lists/numpy by creating a new tensor
        return torch.tensor(x, dtype=torch.float64)

    # ---------------------------------------------------------
    # Extract Input & Parameters
    # ---------------------------------------------------------
    is_numpy = isinstance(u, np.ndarray)
    
    # Convert input u to Tensor
    u_t = to_t(u)

    # Extract p_g1 / p_g2 and convert them to Tensors
    if isinstance(params, dict):
        # Handle Dict input
        p_g1 = to_t(params.get("p_g1", []))
        p_g2 = to_t(params.get("p_g2", []))
    else:
        # Handle List/Tensor input
        params_t = to_t(params)
        p_g1 = params_t[0]
        p_g2 = params_t[1]

    # ---------------------------------------------------------
    # Apply Contextual Logic
    # ---------------------------------------------------------
    # u_t is shape [..., d]
    c1 = u_t[..., 0]
    c5 = u_t[..., 4] 
    
    # Binary mask: 1.0 if c1 != 0
    c1_bin = (c1 != 0).to(u_t.dtype)
    
    # Group 1 values (c1 == 0)
    mu_a0_g1 = p_g1[0] + p_g1[1] * c5
    mu_a1_g1 = p_g1[2] + p_g1[3] * c5
    
    # Group 2 values (c1 != 0)
    mu_a0_g2 = p_g2[0] + p_g2[1] * c5
    mu_a1_g2 = p_g2[2] + p_g2[3] * c5
    
    # Combine
    mu_a0 = mu_a0_g1 * (1.0 - c1_bin) + mu_a0_g2 * c1_bin
    mu_a1 = mu_a1_g1 * (1.0 - c1_bin) + mu_a1_g2 * c1_bin
    
    # ---------------------------------------------------------
    # Rare Tail-Risk Bump (applies to both actions)
    # ---------------------------------------------------------
    HOT_C2 = 1 # as the contexts are normalized c2_unnormalized = 4 corresponds to c2_normalized = 1
    HOT_C3 = 1 # as c3 is binary c3_unnormalized = 1 corresponds to c3_normalized = 1
    HOT_C4 = 1 # as c4 is binary c4_unnormalized = 1 corresponds to c4_normalized = 1
    HOT_C6 = 1 # as the contexts are normalized c6_unnormalized = 3 corresponds to c3_normalized = 1
    TAIL_AMP = torch.tensor(12.0, dtype=mu_a0.dtype, device=mu_a0.device)

    c5_center = torch.tensor(2.5, dtype=mu_a0.dtype, device=mu_a0.device)
    c5_scale = torch.tensor(2.5, dtype=mu_a0.dtype, device=mu_a0.device)
    bump_shape = 0.2 + ((c5 - c5_center) / c5_scale) ** 2

    tail_mask = (
        (u_t[..., 1] == HOT_C2)
        & (u_t[..., 2] == HOT_C3)
        & (u_t[..., 3] == HOT_C4)
        & (u_t[..., 5] == HOT_C6)
    )
    tail_mask_f = tail_mask.to(dtype=mu_a0.dtype)
    tail_bump = TAIL_AMP * bump_shape * tail_mask_f

    mu_a0 = mu_a0 + tail_bump
    mu_a1 = mu_a1 + tail_bump

    # ---------------------------------------------------------
    # Apply Test Shift
    # ---------------------------------------------------------
    if Test and shift is not None:
        shift_t = to_t(shift)
        mu_a0 = mu_a0 + shift_t[0]
        mu_a1 = mu_a1 + shift_t[1]

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------
    res = torch.stack([mu_a0, mu_a1], dim=-1)

    if is_numpy:
        return res.detach().numpy()
        
    return res
