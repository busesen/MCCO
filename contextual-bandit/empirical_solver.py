import numpy as np
import scipy.optimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from costs import cost_ymean

def solve_exact_robust_problem(data_env, x_initial):
    """
    Solve the finite-population robust optimization problem exactly with L-BFGS-B
    and return the optimal parameters and objective value.
    """
    df = data_env["data"]
    ctx_cols = data_env["CTX_COLS"]
    U_all = df[ctx_cols].values
    counts = df["COUNT_1"].values
    N_total = len(U_all)
    
    # Active set optimization
    mask_active = (counts > 0)
    U_outer = U_all[mask_active]
    weights_active = counts[mask_active]
    W_outer = weights_active / weights_active.sum()
    N_active = len(U_outer)
    print(f"Exact Solver: Optimizing over {N_active} active rows (out of {N_total}).")

    # Cost all
    M_all = cost_ymean(U_all, data_env["cost_params"], data_env["shift"], Test=False)
    
    D2_rect = cdist(U_outer, U_all, metric='sqeuclidean')
    mask_conscious_all = (U_all[:, 0] > 0.01) # or equivalently "U_all[:, 0] is not equal to 0"
    mask_not_conscious_all = ~mask_conscious_all

    r_y, r_c, mu = float(data_env["r_y"]), float(data_env["r_c"]), float(data_env["mu"])
    gamma_1 = float(data_env["gamma_1"])
    gamma_2 = float(data_env["gamma_2"])

    def exact_loss(x):
        lam, th1, th2 = x
        p_a1 = np.zeros(N_total)
        p_a1[mask_conscious_all] = th1
        p_a1[mask_not_conscious_all] = th2
        
        V = p_a1 * M_all[:, 0] + (1 - p_a1) * M_all[:, 1]
        
        E_mat = mu * (V.reshape(1, N_total) - lam * D2_rect)
        lse_i = logsumexp(E_mat, axis=1) - np.log(N_total)
        
        reg = gamma_1 * (th1 ** 2) + gamma_2 * (th2 ** 2)
        return np.dot(W_outer, lse_i) / mu + r_y + (r_c**2)*lam + reg

    res = scipy.optimize.minimize(
        exact_loss, x0=np.array(x_initial), bounds=[(0, None), (0, 1), (0, 1)],
        method='L-BFGS-B', options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
    )
    return {"x_opt": res.x, "fun": res.fun, "success": res.success, "message": res.message}
