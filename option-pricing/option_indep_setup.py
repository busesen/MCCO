import numpy as np

# ===============================================================
#   SIMULATOR 
# ===============================================================
def make_simulator(dimension):

    def simulator_gaussian(num_samples):
        """
        Simulates a 1-dimensional standard Normal
        """
        return np.random.normal(0, 1, (num_samples,dimension))
    return simulator_gaussian
# ===============================================================
#   PAYOFF 
# ===============================================================
def make_payoffs(): 
    """
    Returns (func, func_last) for optimal stopping.
    func: used for all but last stage
    func_last: used only for terminal stage
    """

    def func(traj, continuation_sample):
        return max(np.mean(traj[-1]), continuation_sample)

    def func_last(traj):
        return  np.mean(traj[-1])

    return func, func_last
