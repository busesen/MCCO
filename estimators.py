# estimators.py
import numpy as np
# ---------------------------------------------------------------------
#              COST FUNCTIONS
# ---------------------------------------------------------------------
def cost_untruncated(rates, T: int) -> float:
    """
    Computes the expected cost of one tree: 
    prod_{t=1}^{T-1} sum_{l=0}^{infty} 2^l q_t(l)

    Args:
    rate (list or array): rate param r [r1, r2]
    T (int): - time horizon

    Returns:
    float: The expected number of samples per tree for Untruncated MLMC
    """
    product = 1.0
    for r in rates:
        if not (0.5 < r < 1.0):
            raise ValueError(f"rate_untruncated must be in (0.5, 1), got {r}.")
        product *= r / (2 * r - 1)
    return float(product)

def cost_truncated(truncations, rates, T):
    """
    Computes the expected cost of one tree: 
    prod_{t=1}^{T-1} sum_{l=0}^{M_t} 2^l q_t(l)
    
    Args:
    truncation: list, truncation levels of geometric distribution [M1, M2]
    rate (list or array): rate param r [r1, r2]
    T (int): - time horizon

    Returns:
    float: The expected number of samples per tree for Truncated MLMC
    """
    product = 1
    for i in range(T-1):
        r = rates[i]
        M = truncations[i]
        if r == 0.5:
            cost = (M + 1) * r / (1 - (1 - r) ** (M + 1))
        else:
            cost = r / (1 - (1 - r) ** (M + 1)) * (1 - (2 - 2 * r) ** (M + 1)) / (2 * r - 1)
        product *= cost   
    return product  

def cost_saa(sample_sizes, T: int) -> float:
    """
    Computes the cost of one tree (n_1: number of trees)
    prod_{t=1}^{T-1} n_{t+1}

    Args: 
    sample_sizes: n_2, ..., n_T 

    Returns:
    float: The number of samples per tree for SAA
    """
    product = 1.0
    for s in sample_sizes:
        product *= float(s)
    return float(product)
# ---------------------------------------------------------------------
#              ESTIMATORS
# ---------------------------------------------------------------------
def truncated_geometric(r, truncation_point):
    """
    Parameters:
    - rate r (float): The probability of success in each trial (must be between 0 and 1).
    - truncation M (int): The maximum value (inclusive) that can be sampled
    
    Returns:
    - int: A sample from the truncated geometric distribution.
    """
    while True:
        sample = np.random.geometric(r) - 1  # np.random.geometric starts from l=1
        if sample <= truncation_point:
            return sample   

def Truncated_MLMC(simulator, trajectory, truncation, rate, funs, time_horizon):
    """
    simulator: simulator that outputs samples of xi_t
    trajectory: list, the trajectory of process history, default is empty
    truncation: list, truncation levels of geometric distribution [M1, M2]
    rate: list, the parameters of the geometric distribution at each stage [r1, r2]
    funs: list, the functions of interest at each stage
    time_horizon: T 
    Returns: scalar, a biased estimator of F(x)
    """
    # Get the first function
    func = funs[0]
    # Append x to the trajectory
    x = simulator(trajectory, num_samples=1)
    trajectory.append(x)

    
    # If already reached the last depth (horizon = 1)
    if time_horizon == 1:
        res = func(trajectory)
        return  res
    l = truncated_geometric(rate[0], truncation[0])   
    normalization = 1 - (1 - rate[0]) ** (truncation[0] + 1)
    q = rate[0]*(1 - rate[0]) ** l / normalization
    
    funs = funs[1:]  # Remove the first function from the list
    truncation = truncation[1:]  # Remove the first parameter from the list
    rate = rate[1:]
    
    # Recursive call with new list of functions, new parameters, and one less time horizon
    samples = [Truncated_MLMC(simulator, list(trajectory), truncation, rate, funs, time_horizon - 1) for _ in range(2**l)]
    
    
    if l == 0:
        return float(func(trajectory, float(samples[0]) )) / q 
    else:
        # Split samples into even and odd terms
        samples_odd = samples[::2]
        samples_even = samples[1::2]
        mean_odd = float(np.mean(samples_odd))
        mean_even = float(np.mean(samples_even))
        mean_all = 0.5 * mean_odd + 0.5 * mean_even
        estimator_all = func(trajectory, mean_all)
        estimator_odd = func(trajectory, mean_odd)
        estimator_even = func(trajectory, mean_even)
        Delta = estimator_all - (estimator_odd + estimator_even) / 2
        return  float(Delta) / q
    
def Untruncated_MLMC(simulator, trajectory, rate, funs, time_horizon):
    """
    simulator: simulator that outputs samples of xi_t
    trajectory: list, the trajectory of process history, default is empty
    rate: list, the parameters of the geometric distribution at each stage [r1, r2]
    funs: list, the functions of interest at each stage
    time_horizon: T 
    Returns: scalar, an unbiased estimator of F(x)
    """
    
    # Get the first function
    func = funs[0]
    # Append x to the trajectory
    x = simulator(trajectory, num_samples=1)
    trajectory.append(x)
    
    # If already reached the last depth (horizon = 1)
    if time_horizon == 1:
        res = func(trajectory)
        return res
    
    l = np.random.geometric(p=rate[0]) - 1 
    
    num_samples = 2**l
    q = (1 - rate[0]) ** l * rate[0]  # PMF of lambda
    rate = rate[1:]  # Remove the first parameter from the list
    funs = funs[1:]  # Remove the first function from the list
    # Recursive call with new list of functions, new parameters, and one less time horizon
    samples = [Untruncated_MLMC(simulator, list(trajectory), rate, funs, time_horizon - 1) for _ in range(num_samples)]
    
    if l == 0:
        return func(trajectory, samples[0]) / q 
    else:
        # Split samples into even and odd terms
        samples_odd = samples[::2]
        samples_even = samples[1::2]
        mean_odd = np.mean(samples_odd)
        mean_even = np.mean(samples_even)
        mean_all = 0.5 * mean_odd + 0.5 * mean_even
        estimator_all = func(trajectory, mean_all)
        estimator_odd = func(trajectory, mean_odd)
        estimator_even = func(trajectory, mean_even)
        Delta = estimator_all - (estimator_odd + estimator_even) / 2
        return Delta / q

# -------------------
#    SAA
# -------------------
def SAA(simulator, trajectory, inner_sample_size, funs, time_horizon):
    """
    SAA
    """
    func = funs[0]
    x = simulator(trajectory, num_samples=1)
    trajectory.append(x)

    # If already reached the last depth (horizon = 1)
    if time_horizon == 1:
        res = func(trajectory)
        return  res
 
    num_samples = inner_sample_size[0]
    
    funs = funs[1:]  
    inner_sample_size = inner_sample_size[1:]  
    samples = [SAA(simulator, list(trajectory), inner_sample_size, funs, time_horizon - 1) for _ in range(num_samples)]
    # mean_all = np.mean([x.item() for arr in samples for x in arr]) 
    mean_all = np.mean(samples)
    return  func(trajectory, mean_all)