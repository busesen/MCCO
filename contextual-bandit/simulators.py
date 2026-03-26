import torch
from typing import Dict, Any
from costs import cost_ymean

def simulator(data_env: Dict[str, Any], trajectory, x: torch.Tensor, time: int, num_samples: int):
    # Time 3: Outer Context (Weighted)
    if time == 3:
        return data_env["data_tensor"][torch.multinomial(data_env["weights_prob"], num_samples, replacement=True)]
    
    # Time 2: Inner Context (Uniform)
    if time == 2:
        # Trajectory here is 'xi1', which has shape (sub_B, dim)
        # We need to generate 'num_samples' (n_u) for EACH item in the batch.
        # Output shape must be: (sub_B, num_samples, dim)
        
        # Detect Batch Size (sub_B)
        if isinstance(trajectory, torch.Tensor):
            sub_B = trajectory.shape[0]
        else:
            # Fallback if trajectory is a list or empty (shouldn't happen)
            sub_B = 1 
            
        # Generate total required indices (sub_B * n_u)
        total_samples = sub_B * num_samples
        indices = torch.randint(0, data_env["data_tensor"].shape[0], (total_samples,))
        
        # Sample from data tensor
        samples = data_env["data_tensor"][indices] # Shape: (sub_B * n_u, dim)
        
        # Reshape to (sub_B, n_u, dim) so the MLMC code can broadcast correctly
        return samples.view(sub_B, num_samples, -1)

    # Time 1: Costs 
    u_tensor = trajectory[-1]
    target_mean = cost_ymean(u_tensor, data_env["cost_params"])

    sigma_sq = torch.diagonal(data_env["covariance"]).to(
        dtype=target_mean.dtype, device=target_mean.device
    )
    mu = torch.log(target_mean) - 0.5 * sigma_sq
    chol = data_env["chol_cov"].to(dtype=target_mean.dtype, device=target_mean.device)

    if u_tensor.ndim == 3:
        # Vectorized SAA path: u is (B, n0, d), return (B, n0, n1, 2)
        B, n0 = u_tensor.shape[0], u_tensor.shape[1]
        Z = torch.randn(B, n0, num_samples, 2, dtype=target_mean.dtype, device=target_mean.device)
        log_y = mu.unsqueeze(2) + (Z @ chol.T)
        return torch.exp(log_y)

    # Scalar/non-vectorized path: u is (n0, d), return (n0, n1, 2)
    Z = torch.randn(u_tensor.shape[0], num_samples, 2, dtype=target_mean.dtype, device=target_mean.device)
    log_y = mu.unsqueeze(1) + (Z @ chol.T)
    return torch.exp(log_y)

