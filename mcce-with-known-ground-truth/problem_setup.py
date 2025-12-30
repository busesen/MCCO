import numpy as np

def simulator_gaussian_martingale(trajectory, num_samples, mean=np.pi/2):
    # Check if trajectory is empty
    if len(trajectory) == 0:
        # Generate samples from normal distribution with mean = mean and std = 1
        samples = np.random.normal(mean, 1, num_samples)
    else:
        # Generate samples from normal distribution with mean = last element of trajectory and std = 1
        samples = np.random.normal(trajectory[-1], 1, num_samples).reshape(-1)  
    return samples

#### T = 3  ##############################################################
#### the functions are f_1, f_2, f_3 #####################################
#### f_1 (xi_1, x1) = sin(xi_1 + x1) #####################################
#### f_2 (xi_2, x2) = sin(xi_2 - x1) #####################################
#### f_3 (xi_3, x3) = xi_3 ###############################################
#### xi_1 ~ N(pi/2,1),   xi_2 ~ N(xi_1,1),   xi_3~ N(xi_2,1) #############
#### Ground truth = np.exp(-1/2) = 0.6065306597126334 ####################

true_val = np.exp(-1/2)

def func1(traj, sample):
    # traj[-1]: xi_1, sample: x1
    return np.sin(traj[-1] + sample)
def func2(traj, sample):
    # traj[-1]: xi_2, sample: x2
    return np.sin(traj[-1] - sample)
def func3(traj):
    # traj[-1]: xi_3
    return traj[-1]
funs = [func1, func2, func3]