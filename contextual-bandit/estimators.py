import torch
from typing import Tuple
from utils import to_tensor
from costs import cost_ymean
from typing import Tuple, Callable
from simulators import simulator
import numpy as np

@torch.jit.script
def _compute_estimators_jit(
    y_batch: torch.Tensor, 
    p_vec: torch.Tensor, 
    lam: torch.Tensor, 
    dist2: torch.Tensor,
    r_y: float,
    r_c_sq: float, 
    mu: float, 
    m1_vec: torch.Tensor, 
    m2_vec: torch.Tensor, 
    q2: float, 
    l2: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes MLMC estimators at T=2 for a given level l2 (the terms hat{H}_2(x) and hat{G}_2(x)).

    This function computes the scaled versions of the weighting term \hat{H}_2(x) and 
    the gradient term \hat{G}_2(x). The 'scale' information is contained in the 
    returned 'shift_u'.

    The scaling is necessary to avoid numerical explosions (overflow) or vanishing 
    to 0 (underflow) that would otherwise occur when computing differences of 
    exponentials: exp(mean of all) - 0.5exp(mean of even) - 0.5exp(mean of odd).
    
    Args:
        y_batch (Tensor): Simulated cost samples. Shape (Batch, N, 2).
        p_vec (Tensor): Policy probability for action a=0 (theta1 or theta2). Shape (Batch,).
        lam (Tensor): Current lambda value (scalar).
        dist2 (Tensor): Squared distances ||u - c'||^2. Shape (Batch,).
        r_y, r_c_sq, mu: Scalar constants from the problem formulation.
        m1_vec, m2_vec (Tensor): Gradient masks for theta1/theta2 logic. Shape (Batch,).
        q2 (float): Probability of sampling this inner level (for importance sampling correction).
        l2 (int): The current value of level at T=2  (l_2 = 0, 1, 2, ..., M_2).

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - shifts (Tensor): The logarithmic scale factor (max exponent).
            - H2scaled (Tensor): The scaled probability weight, i.e., H2scaled = H2 * exp(-shift_u)   (Batch,).
            - G2scaled (Tensor): The scaled gradient term, i.e., G2scaled = G2 * exp(-shift_u)  (Batch, 3).
    """  
    # Gradient of f3 w.r.t lambda: (r_c^2 - dist^2)  
    dlam = r_c_sq - dist2
    # Compute mean over all 2^l2 inner samples
    mean_y = y_batch.mean(dim=1)

    # Difference in means between action a0 and a1
    ydiff = mean_y[:, 0] - mean_y[:, 1]

    # The linear term inside the exponent for the fine level
    # A = hat{E}^{ell_2}_2[f3] = mu * (Expected_Cost + Regularization)
    mean_f3 = p_vec * mean_y[:, 0] + (1.0 - p_vec) * mean_y[:, 1] + r_y + r_c_sq * lam - lam * dist2

    # Gradients of A w.r.t [lambda, theta1, theta2]
    # d(A)/d(lam)   = dlam
    # d(A)/d(theta) = ydiff * mask (since p_vec is theta1 or theta2)
    A = mu * mean_f3
    G3_all = torch.stack([dlam, ydiff * m1_vec, ydiff * m2_vec], dim=1) # hat E_2^{ell_2}[hat G_3(x)]

    if l2 == 0: 
        # At level 0, there is no coarse level to subtract.
        return A, (torch.ones_like(A) / q2), (G3_all / q2)

    y_even, y_odd = y_batch[:, 0::2, :], y_batch[:, 1::2, :]
    # Compute means for the two coarse sub-groups
    mean_even, mean_odd = y_even.mean(dim=1), y_odd.mean(dim=1)
    
    f3_even = p_vec * mean_even[:, 0] + (1.0 - p_vec) * mean_even[:, 1] + r_y + r_c_sq * lam - lam * dist2
    f3_odd = p_vec * mean_odd[:, 0] + (1.0 - p_vec) * mean_odd[:, 1] + r_y + r_c_sq * lam - lam * dist2
    # Compute exponents (B_even and B_odd) for the coarse level estimators
    # Note: We reuse the formula for 'mean_f3' but with coarse means.
    Be, Bo = mu * f3_even, mu * f3_odd # hat{E}_2^{ell_2, e}[f_3] and hat{E}_2^{ell_2, o}[f_3]  (f_3=\hat{H}_3)

    # Gradients for the coarse terms
    ydiff_even, ydiff_odd = mean_even[:, 0] - mean_even[:, 1], mean_odd[:, 0] - mean_odd[:, 1]
    G3_even = torch.stack([dlam, ydiff_even * m1_vec, ydiff_even * m2_vec], dim=1)  # hat E_2^{ell_2, e}[G_3(x)]
    G3_odd = torch.stack([dlam, ydiff_odd * m1_vec, ydiff_odd * m2_vec], dim=1)  # hat E_2^{ell_2, o}[G_3(x)]

    # -----------------------------------------------------------------------
    #  LOG-SUM-EXP STABILIZATION & COMBINATION
    # -----------------------------------------------------------------------
    # We need to compute: 
    #   Numerator = exp(A)*Grad(A) - 0.5 * ( exp(Be)*Grad(Be) + exp(Bo)*Grad(Bo) )
    #   Denominator term handled later via H2scaled.
    maxB = torch.maximum(Be, Bo)
    # Compute stable exponentials relative to maxB
    s_even, s_odd = torch.exp(Be - maxB), torch.exp(Bo - maxB)
    denom = s_even + s_odd
    # The coarse gradient is the weighted average of the even/odd gradients
    grad_mix = (s_even.unsqueeze(1) * G3_even + s_odd.unsqueeze(1) * G3_odd) / denom.unsqueeze(1)

    # Combine Fine (A) and Coarse (maxB) logic
    shift_u = torch.maximum(A, maxB)
    eA, eB = torch.exp(A - shift_u), torch.exp(maxB - shift_u) * 0.5 * denom
    # return shifts, H2scaled and G2scaled
    return shift_u, (eA - eB) / q2, (eA.unsqueeze(1) * G3_all - eB.unsqueeze(1) * grad_mix) / q2

@torch.jit.script
def _ratio_jit(shifts: torch.Tensor, H2: torch.Tensor, G2: torch.Tensor, eps: float) -> torch.Tensor:
    r"""
    Computes the MLMC gradient estimator term \hat{g}_1^\ell(x) (or its coarse counterparts 
    \hat{g}_1^{\ell, e}(x), \hat{g}_1^{\ell, o}(x)).

    Context:
        The objective is J(x) = E[ 1/mu * log( Z(x) ) ], where Z(x) is an inner expectation.
        By the chain rule, grad(J) = E[ 1/(mu * Z(x)) * grad(Z(x)) ].
        
        This function computes that ratio:
            \hat{g} = Numerator / Denominator
        where:
            Numerator   corresponds to \hat{E}_1^\ell[ \hat{G}_2(x) ] (grad of inner term)
            Denominator corresponds to \hat{E}_1^\ell[ \hat{H}_2(x) ] (the inner term Z(x))

    Args:
        shifts (Tensor): Log-values of the inner exponents (used for stability). Shape (N,).
        H2 (Tensor): Estimators for the inner integral Z(x) (unnormalized). Shape (N,).
        G2 (Tensor): Estimators for the gradient of the inner integral (unnormalized). Shape (N, D).
        eps (float): Stability constant.

    Returns:
        Tensor: The computed gradient estimator \hat{g}_1^\ell(x). Shape (D,).
    """
    max_s = shifts.max()
    w = torch.exp(shifts - max_s)
    w = w / w.sum()
    denom = (w * H2).sum()
    numer = (w.unsqueeze(1) * G2).sum(dim=0)
    scale = (w * H2.abs()).sum()
    if torch.abs(denom) < eps * scale:
        denom = torch.sign(denom) * eps
        if denom == 0.0: denom = torch.tensor(eps, dtype=shifts.dtype)
    # Apply Chain Rule: grad(log(Z)) = grad(Z) / Z
    return numer / denom

@torch.jit.script
def _ratio_jit_batched(
    shifts: torch.Tensor, 
    H2: torch.Tensor, 
    G2: torch.Tensor, 
    eps: float
) -> torch.Tensor:
    r"""
    Vectorized Ratio Estimator.
    Input shapes:
      shifts: (B, N)
      H2:     (B, N)
      G2:     (B, N, D)
    Returns:
      g:      (B, D)
    """
    # 1. Stable Weights (Softmax-like) along dim 1 (the N samples)
    max_s = shifts.max(dim=1, keepdim=True)[0]
    w = torch.exp(shifts - max_s)
    w_sum = w.sum(dim=1, keepdim=True)
    w = w / w_sum

    # 2. Denominator: Z(x) = sum(w * H)
    denom = (w * H2).sum(dim=1)  # (B,)

    # 3. Numerator: grad(Z) = sum(w * G)
    # w: (B, N) -> (B, N, 1) to broadcast against G2: (B, N, D)
    numer = (w.unsqueeze(2) * G2).sum(dim=1) # (B, D)

    # 4. Stability Check
    scale = (w * H2.abs()).sum(dim=1)
    
    # Create mask for unstable denominators
    unstable = torch.abs(denom) < (eps * scale)
    
    # Safe denominator (vectorized sign logic)
    denom_safe = torch.where(denom >= 0, torch.tensor(eps), torch.tensor(-eps))
    # If denominator was exactly 0, sign might be 0, fix that
    denom_safe = torch.where(denom_safe == 0, torch.tensor(eps), denom_safe)
    
    final_denom = torch.where(unstable, denom_safe.to(denom.dtype), denom)
    
    # 5. Ratio
    return numer / final_denom.unsqueeze(1)

# def truncated_MLMC_grad(
#     x_initial, simulator, truncation_point, rate, data_env,
#     *, eps_denom: float = 1e-6, clip_lambda: float = 0.0, clip_thetas: float = 0.0
# ):
#     r"""
#     Computes the truncated MLMC gradient estimator \hat{G}(x) for the nested DRO objective
#     and the total computational cost (number of \xi_3 samples generated), not the 'expected' cost.
       
#     Args:
#         x_initial (Tensor): Current optimization parameters [lambda, theta1, theta2].
#         simulator (Callable): Function to simulate contexts and costs.
#         truncation_point (tuple): Max truncation levels (M1, M2) for outer and inner loops.
#         rate (tuple): Geometric rates (r1, r2) for probability of levels.
#         data_env (dict): Environment constants (covariance, r_c, etc.).
#         eps_denom (float): Stability term for the ratio estimator.
#         clip_lambda (float): Max absolute gradient for lambda.
#         clip_thetas (float): Max L2 norm for theta gradients.

#     Returns:
#         tuple: (Gradient Tensor [3], Sample Cost [float])
#     """
#     # ---------------------------------------------------------
#     #  Setup 
#     # ---------------------------------------------------------
#     x = to_tensor(x_initial).detach().clone()
#     lam = x[0]
#     r_y, r_c, mu = data_env["r_y"], data_env["r_c"], data_env["mu"]
#     r1, r2 = rate[0], rate[1]
#     M1, M2 = int(truncation_point[0]), int(truncation_point[1])
    
#     # Pre-compute Cholesky for fast sampling of costs y ~ N(mean, Cov)
#     cov, cost_params = data_env["covariance"], data_env["cost_params"]
#     L_chol = data_env.get("chol_cov", torch.linalg.cholesky(cov)).to(dtype=x.dtype)

#     # ---------------------------------------------------------
#     #  Outer Level Sampling (Level l1)
#     # ---------------------------------------------------------
#     l1 = torch.multinomial(torch.softmax(torch.log(torch.tensor(r1)) + torch.arange(M1+1)*torch.log(torch.tensor(1-r1)), 0), 1).item()
#     q1 = (r1 * (1-r1)**l1) / (1 - (1-r1)**(M1+1))
#     n_u = 2 ** l1

#     # xi1: The center context c' 
#     # xi2 = u_rows: The batch of contexts u sampled from Uniform(C)
#     xi1 = simulator(data_env, [], x, 3, 1)
#     u_rows = simulator(data_env, xi1, x, 2, n_u)
    
#     # Compute distances and cost means for these contexts ||u-c'||^2
#     dist2 = ((u_rows - xi1[0].unsqueeze(0))**2).sum(dim=1)
#     u_means = cost_ymean(u_rows, cost_params)

#     shifts = torch.empty(n_u, dtype=x.dtype)
#     H2s = torch.empty(n_u, dtype=x.dtype)
#     G2s = torch.empty(n_u, 3, dtype=x.dtype)
    
#     # We will count the number of paths generated at the scenario tree 
#     sample_cost = 0.0
    
#     # ---------------------------------------------------------
#     #  Inner Level Sampling (Level l2) - Vectorized
#     # ---------------------------------------------------------
#     probs_l2 = torch.softmax(torch.log(torch.tensor(r2)) + torch.arange(M2+1)*torch.log(torch.tensor(1-r2)), 0)
#     l2_all = torch.multinomial(probs_l2, n_u, replacement=True)
    
#     for l2_t in torch.unique(l2_all):
#         l2 = int(l2_t.item())
#         n_y = 2 ** l2
#         q2 = (r2 * (1-r2)**l2) / (1 - (1-r2)**(M2+1))
#         idxs = torch.where(l2_all == l2)[0]
        
#         B = len(idxs) # Indices of contexts that fell into this level
#         sample_cost += float(n_y * B) # count the number of samples generated

#         # Fast Simulation: y = mean + Z * L_chol^T
#         y_b = (torch.randn(B, n_y, 2, dtype=x.dtype) @ L_chol.T) + u_means[idxs].unsqueeze(1)
        
#         # Determine Policy parameters for these contexts (based on group c0)
#         use_th2 = (u_rows[idxs, 0] <= 0.01)
#         p_v = torch.where(use_th2, x[2], x[1])
#         # Masks for gradients w.r.t theta1 vs theta2
#         m1 = torch.where(use_th2, torch.tensor(0.0), torch.tensor(1.0))
#         m2 = torch.where(use_th2, torch.tensor(1.0), torch.tensor(0.0))

#         # Compute Raw Inner Estimators (JIT Compiled Kernel)
#         # Returns shift, scaled H (denom), and scaled G (numer)
#         shifts[idxs], H2s[idxs], G2s[idxs] = _compute_estimators_jit(y_batch=y_b, p_vec=p_v, lam=lam, dist2=dist2[idxs], r_y=float(r_y), r_c_sq=float(r_c)**2, 
#                                           mu=float(mu), m1_vec=m1, m2_vec=m2, q2=float(q2), l2=l2)
#     # ---------------------------------------------------------
#     # Construct Final Estimator 
#     # ---------------------------------------------------------
#     # Aggregate ALL samples to form the "Fine" estimator (Level l1)
#     g1_all = _ratio_jit(shifts, H2s, G2s, eps_denom)
#     if l1 == 0:
#         G_hat = g1_all / q1
#     else:
#         g1_e = _ratio_jit(shifts[0::2], H2s[0::2], G2s[0::2], eps_denom)
#         g1_o = _ratio_jit(shifts[1::2], H2s[1::2], G2s[1::2], eps_denom)
#         G_hat = (g1_all - 0.5 * (g1_e + g1_o)) / q1

#     # ---------------------------------------------------------
#     #  Gradient Clipping
#     # ---------------------------------------------------------
#     g_lam, g_th= G_hat[0], G_hat[1:]
#     if clip_lambda > 0 and torch.abs(g_lam) > clip_lambda:
#         g_lam = torch.sign(g_lam) * clip_lambda
#     if isinstance(clip_thetas, list):
#         # Case: Separate clipping for theta1 and theta2
#         for i in range(len(g_th)):
#             c_val = clip_thetas[i]
#             if c_val > 0 and torch.abs(g_th[i]) > c_val:
#                 g_th[i] = torch.sign(g_th[i]) * c_val
                
#     elif clip_thetas > 0:
#         # Case: joint L2 norm clipping (if a single float is passed)
#         nrm = torch.norm(g_th)
#         if nrm > clip_thetas: 
#             g_th = g_th * (clip_thetas / nrm)
            
#     return torch.cat([g_lam.unsqueeze(0), g_th]).detach(), float(sample_cost)

def truncated_MLMC_grad_vectorized(
    x_initial, simulator, truncation_point, rate, data_env,
    *, eps_denom: float = 1e-6, clip_lambda: float = 0.0, clip_thetas: float = 0.0,
    batch_size: int = 1
):
    x = torch.as_tensor(x_initial).detach().clone()
    lam = x[0]
    r_y, r_c, mu = data_env["r_y"], data_env["r_c"], data_env["mu"]
    r1, r2 = rate[0], rate[1]
    M1, M2 = int(truncation_point[0]), int(truncation_point[1])
    
    cov, cost_params = data_env["covariance"], data_env["cost_params"]
    # Ensure Cholesky is on correct device/dtype
    L_chol = data_env.get("chol_cov", torch.linalg.cholesky(cov)).to(dtype=x.dtype, device=x.device)

    # ---------------------------------------------------------
    # 1. Vectorized Outer Level Sampling (l1)
    # ---------------------------------------------------------
    # Probabilities for levels 0..M1
    p1_logits = torch.log(torch.tensor(r1)) + torch.arange(M1+1)*torch.log(torch.tensor(1-r1))
    p1 = torch.softmax(p1_logits, 0)
    
    # Sample l1 for the entire batch at once
    l1_all = torch.multinomial(p1, batch_size, replacement=True) # (Batch,)

    # Initialize accumulators
    total_grad = torch.zeros_like(x)
    total_cost = 0.0

    # ---------------------------------------------------------
    # 2. Group by Level l1
    # ---------------------------------------------------------
    unique_l1 = torch.unique(l1_all)
    
    for l1_val in unique_l1:
        l1 = int(l1_val.item())
        # Indices in the batch that define this group
        group_mask = (l1_all == l1)
        sub_B = int(group_mask.sum().item())
        
        q1 = (r1 * (1-r1)**l1) / (1 - (1-r1)**(M1+1))
        n_u = 2 ** l1
        
        # -----------------------------------------------------
        # 3. Simulate Outer Contexts (Vectorized for sub-batch)
        # -----------------------------------------------------
        # xi1: (sub_B, dim)
        xi1 = simulator(data_env, [], x, 3, sub_B)
        
        # u_rows: (sub_B, n_u, dim)
        u_rows = simulator(data_env, xi1, x, 2, n_u)
        
        # Distances: (sub_B, n_u)
        dist2 = ((u_rows - xi1.unsqueeze(1))**2).sum(dim=2)
        
        # Means: (sub_B, n_u, 2) - Assuming cost_ymean handles batching
        # If cost_ymean expects (N, dim), we flatten and reshape
        flat_u = u_rows.view(-1, u_rows.shape[-1])
        u_means_flat = cost_ymean(flat_u, cost_params)
        u_means = u_means_flat.view(sub_B, n_u, 2)
        
        # -----------------------------------------------------
        # 4. Inner Level Sampling (l2)
        # -----------------------------------------------------
        # Total contexts to simulate = sub_B * n_u
        total_contexts = sub_B * n_u
        
        p2_logits = torch.log(torch.tensor(r2)) + torch.arange(M2+1)*torch.log(torch.tensor(1-r2))
        p2 = torch.softmax(p2_logits, 0)
        
        # Sample l2 for EVERY context in this sub-batch
        l2_all_inner = torch.multinomial(p2, total_contexts, replacement=True)
        
        # Containers for results of _compute_estimators_jit
        # We need to store them to recombine later. 
        # Shape: (sub_B * n_u, ...) -> reshape to (sub_B, n_u, ...) later
        shifts_flat = torch.zeros(total_contexts, dtype=x.dtype, device=x.device)
        H2s_flat    = torch.zeros(total_contexts, dtype=x.dtype, device=x.device)
        G2s_flat    = torch.zeros(total_contexts, 3, dtype=x.dtype, device=x.device)
        
        unique_l2 = torch.unique(l2_all_inner)
        
        for l2_inner_val in unique_l2:
            l2_in = int(l2_inner_val.item())
            idx_in = torch.where(l2_all_inner == l2_in)[0]
            count_in = len(idx_in)
            
            n_y = 2 ** l2_in
            q2 = (r2 * (1-r2)**l2_in) / (1 - (1-r2)**(M2+1))
            
            total_cost += float(n_y * count_in)
            
            # -------------------------------------------------
            # 5. Inner Simulation (Vectorized)
            # -------------------------------------------------
            # Generate y: (count_in, n_y, 2)
            # Z: (count_in, n_y, 2)
            Z = torch.randn(count_in, n_y, 2, dtype=x.dtype, device=x.device)
            # Apply covariance: Z @ L.T
            # L_chol: (2,2). Z: (B, N, 2). Result: (B, N, 2)
            noise = Z @ L_chol.T 
            
            # Get means for these specific contexts
            # u_means was (sub_B, n_u, 2), flattened to (total_contexts, 2)
            u_means_active = u_means.view(-1, 2)[idx_in] # (count_in, 2)
            
            y_b = noise + u_means_active.unsqueeze(1)
            
            # Policy logic
            # flat_u was (total_contexts, dim)
            u_active = flat_u[idx_in]
            use_th2 = (u_active[:, 0] <= 0.01)
            p_v = torch.where(use_th2, x[2], x[1])
            m1 = torch.where(use_th2, torch.tensor(0.0), torch.tensor(1.0))
            m2 = torch.where(use_th2, torch.tensor(1.0), torch.tensor(0.0))
            
            dist2_active = dist2.view(-1)[idx_in]
            
            # -------------------------------------------------
            # 6. JIT Kernel Call
            # -------------------------------------------------
            s_out, h_out, g_out = _compute_estimators_jit(
                y_batch=y_b, p_vec=p_v, lam=lam, dist2=dist2_active,
                r_y=float(r_y), r_c_sq=float(r_c)**2, mu=float(mu),
                m1_vec=m1, m2_vec=m2, q2=float(q2), l2=l2_in
            )
            
            shifts_flat[idx_in] = s_out
            H2s_flat[idx_in] = h_out
            G2s_flat[idx_in] = g_out

        # -----------------------------------------------------
        # 7. Aggregate MLMC Outer Level
        # -----------------------------------------------------
        # Reshape results back to (sub_B, n_u)
        shifts = shifts_flat.view(sub_B, n_u)
        H2s = H2s_flat.view(sub_B, n_u)
        G2s = G2s_flat.view(sub_B, n_u, 3)
        
        # Calculate Ratio Estimators using Batched Helper
        # Result: (sub_B, 3)
        g1_all = _ratio_jit_batched(shifts, H2s, G2s, eps_denom)
        
        if l1 == 0:
            G_hat = g1_all / q1
        else:
            # Slices: 0::2 vs 1::2 along dim 1 (n_u)
            g1_e = _ratio_jit_batched(shifts[:, 0::2], H2s[:, 0::2], G2s[:, 0::2], eps_denom)
            g1_o = _ratio_jit_batched(shifts[:, 1::2], H2s[:, 1::2], G2s[:, 1::2], eps_denom)
            G_hat = (g1_all - 0.5 * (g1_e + g1_o)) / q1
            
        # Accumulate gradients (sum them up, we divide by B at the end)
        total_grad += G_hat.sum(dim=0)
        
    # Average over batch
    avg_grad = total_grad / float(batch_size)
    
    # ---------------------------------------------------------
    # 8. Clipping (Same as before)
    # ---------------------------------------------------------
    g_lam, g_th = avg_grad[0], avg_grad[1:]
    
    if clip_lambda > 0 and torch.abs(g_lam) > clip_lambda:
        g_lam = torch.sign(g_lam) * clip_lambda
        
    if isinstance(clip_thetas, list):
        for i in range(len(g_th)):
            c_val = clip_thetas[i]
            if c_val > 0 and torch.abs(g_th[i]) > c_val:
                g_th[i] = torch.sign(g_th[i]) * c_val
    elif clip_thetas > 0:
        nrm = torch.norm(g_th)
        if nrm > clip_thetas: 
            g_th = g_th * (clip_thetas / nrm)
            
    return torch.cat([g_lam.unsqueeze(0), g_th]).detach(), float(total_cost)

def SAA_grad(x_initial, simulator, data_env, n, batch_size=1):
    """
    SAA Estimator.
    """
    r_y, r_c, mu = data_env["r_y"], data_env["r_c"], data_env["mu"]
    x = torch.as_tensor(x_initial).detach().clone().requires_grad_(True)
    
    # 1. Outer Sampling (Batch Parallel)
    # Simulator must return (Batch, dim)
    xi1 = simulator(data_env, [], x, 3, batch_size) 
    
    # 2. Middle Sampling
    # Simulator returns (Batch, n[0], dim)
    u = simulator(data_env, xi1, x, 2, n[0]) 
    
    # 3. Inner Sampling
    # Simulator returns (Batch, n[0], n[1], dim) usually, or flat. 
    xi3 = simulator(data_env, [xi1, u], x.detach(), 1, n[1]) 
    
    # Policy (pi) calculation
    # dist2: (Batch, n[0])
    dist2 = ((u - xi1.unsqueeze(1))**2).sum(dim=2) 
    
    mask = (u[:, :, 0] <= 0.01).to(x.dtype) # (Batch, n[0])
    
    # pi: (Batch, n[0], 2)
    p_val = mask * x[2] + (1 - mask) * x[1]
    pi = torch.stack([p_val, 1.0 - p_val], dim=2)

    # Scores
    # xi3.mean(dim=2): Average over n[1] -> (Batch, n[0], 2) or (Batch, n[0]) depending on simulator
    xi3_mean = xi3.mean(dim=2) # (Batch, n[0], 2) assuming simulator returns vector costs
    
    # Inner score: dot product of pi and costs: E[y]
    inner_expected = (pi * xi3_mean).sum(dim=2) # (Batch, n[0])
    scores = inner_expected + r_y + (r_c**2)*x[0] - x[0]*dist2 # (Batch, n[0])
    
    # LogSumExp (Vectorized over n[0])
    H = (torch.logsumexp(mu * scores, dim=1) - np.log(n[0])) / mu # (Batch,)
    
    # Autograd
    # We want the mean of gradients, which is equivalent to grad of mean
    loss = H.mean()
    g = torch.autograd.grad(loss, x)[0]
    
    # Scenario count: number of leaf cost realizations only
    num_scenarios = batch_size * n[0] * n[1]
    return g, float(num_scenarios)

Estimator = Callable[[torch.Tensor], Tuple[torch.Tensor, float]]

def _apply_theta_quadratic_grad(g: torch.Tensor, x: torch.Tensor, data_env) -> torch.Tensor:
    """
    Add gradient of gamma_1 * theta1^2 + gamma_2 * theta2^2.
    """
    gamma_1 = float(data_env.get("gamma_1", 0.0))
    gamma_2 = float(data_env.get("gamma_2", 0.0))
    if gamma_1 == 0.0 and gamma_2 == 0.0:
        return g

    g = g.clone()
    x_det = x.detach()
    if g.numel() >= 3 and x_det.numel() >= 3:
        g[1] = g[1] + 2.0 * gamma_1 * x_det[1]
        g[2] = g[2] + 2.0 * gamma_2 * x_det[2]
    return g

def make_estimator_saa(data_env, n_inner) -> Estimator:
    # Now accepts 'batch_size' in the call
    def _est(x, batch_size=1):
        g, c = SAA_grad(x, simulator, data_env, n_inner, batch_size=batch_size)
        return _apply_theta_quadratic_grad(g, x, data_env), c
    return _est

def make_estimator_mlmc(data_env, trunc_M, rate, clip_lam, clip_th) -> Estimator:
    def _est(x, batch_size=1):
        g, c = truncated_MLMC_grad_vectorized(
            x, simulator, trunc_M, rate, data_env,
            clip_lambda=clip_lam, clip_thetas=clip_th,
            batch_size=batch_size
        )
        return _apply_theta_quadratic_grad(g, x, data_env), c
    return _est

