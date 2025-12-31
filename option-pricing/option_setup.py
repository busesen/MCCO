import numpy as np

# ===============================================================
#   SIMULATOR — creates a simulator with given parameters
# ===============================================================
def make_simulator(gamma, delta, sigma, dimension, S0):

    def simulator_geo_brownian(trajectory, num_samples):
        """
        Simulates a m-dimensional Geometric Brownian Motion
        """
        dt = 1.0
        S0_arr = S0 * np.ones((num_samples, dimension))
        dW = np.random.normal(0, np.sqrt(dt), (num_samples, dimension))

        if len(trajectory) == 0:
            return S0_arr

        S_prev = np.tile(np.array(trajectory[-1]).reshape(1, -1), (num_samples, 1))
        drift = (gamma - delta - 0.5 * sigma**2) * dt
        diffusion = sigma * dW
        return S_prev * np.exp(drift + diffusion)

    return simulator_geo_brownian
# ===============================================================
#   PAYOFF — creates payoff functions
# ===============================================================
def make_payoffs(K, gamma): 
    """
    Returns (func, func_last) for Bermudan option.
    func: used for all but last stage
    func_last: used only for terminal stage
    """
    def func(traj, continuation_sample):
        payoff = max(K - np.mean(traj[-1]), 0.0)
        return max(payoff, np.exp(-gamma) * continuation_sample)

    def func_last(traj):
        return max(K - np.mean(traj[-1]), 0.0)

    return func, func_last
