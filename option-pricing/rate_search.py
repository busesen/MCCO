import numpy as np
import matplotlib.pyplot as plt
import json, os, time, argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from matplotlib.ticker import ScalarFormatter
from option_indep_setup import make_simulator, make_payoffs
from estimators import *

# ---------------------------------------------------------------------
#          MATPLOTLIB CONFIGURATION
# ---------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb}\usepackage{times}"
})
# ---------------------------------------------------------------------
#          HELPER: CONVEX OPTIMIZATION & DATA PARSING
# ---------------------------------------------------------------------
def solve_convex_pwl(x_in, y_in):
    """
    Fits a convex piecewise linear function to (x, y).
    Returns (sorted_unique_x, fitted_y).
    """
    x = np.array(x_in)
    y = np.array(y_in)
    
    # Sort and aggregate duplicates (take mean of y for same x)
    ux = np.unique(x)
    uy = np.array([y[x == v].mean() for v in ux])

    # Need at least 3 points to define changing slope
    if len(ux) < 3:
        return ux, uy

    # Objective: Minimize sum of squared errors
    def objective(z):
        return np.sum((z - uy)**2)

    # Constraint: Slope must be non-decreasing (convexity)
    constraints = []
    for i in range(len(ux) - 2):
        def convexity_constraint(z, i=i):
            slope1 = (z[i+1] - z[i]) / (ux[i+1] - ux[i])
            slope2 = (z[i+2] - z[i+1]) / (ux[i+2] - ux[i+1])
            return slope2 - slope1 
        constraints.append({'type': 'ineq', 'fun': convexity_constraint})

    res = minimize(objective, x0=uy, constraints=constraints, tol=1e-6)
    
    if res.success:
        return ux, res.x
    return ux, uy

def parse_results(summary):
    """
    Extracts and normalizes data from the summary JSON dictionary 
    into a structured format for plotting and analysis.
    """
    # Parse Truncated Data
    data_tr = {} # Keyed by truncation level
    if summary.get("trunc_mlmc"):
        for _, info in summary["trunc_mlmc"].items():
            t_lvl = info["trunc"]
            if t_lvl not in data_tr:
                data_tr[t_lvl] = []
            
            data_tr[t_lvl].append({
                "r": info["rate"],
                "time": info["time"],
                "work_norm_var": info["time"] * (info["std"] ** 2)
            })
    
    # Sort lists by rate
    for t_lvl in data_tr:
        data_tr[t_lvl].sort(key=lambda x: x["r"])

    # Parse Untruncated Data
    data_untr = []
    source = summary.get("untrunc_mlmc")
    if source:
        # Handle potential dict wrapper variations
        if isinstance(source, dict) and "rate" not in source:
             for _, info in source.items():
                data_untr.append({
                    "r": info["rate"],
                    "time": info["time"],
                    "work_norm_var": info["time"] * (info["std"] ** 2)
                })
        elif isinstance(source, dict) and "rate" in source:
             data_untr.append({
                "r": source["rate"],
                "time": source["time"],
                "work_norm_var": source["time"] * (source["std"] ** 2)
            })
    
    data_untr.sort(key=lambda x: x["r"])

    return data_tr, data_untr

# ---------------------------------------------------------------------
#          ANALYSIS & PLOTTING FUNCTIONS
# ---------------------------------------------------------------------
def print_optimal_rates(summary, output_file=None):
    """
    Calculates and prints the rate 'r' that minimizes the Work-Normalized Variance.
    """
    data_tr, data_untr = parse_results(summary)
    
    header = f"{'CURVE':<20} | {'MIN (RAW)':<12} | {'MIN (CONVEX)':<12}"
    divider = "-" * len(header)
    
    def log(message):
        print(message)
        if output_file:
            with open(output_file, "a") as f:
                f.write(message + "\n")

    log("\n" + "=" * len(header))
    log(header)
    log(divider)

    def process_and_log(label, dataset):
        if not dataset:
            return
        
        rs = [d["r"] for d in dataset]
        wnv = [d["work_norm_var"] for d in dataset]

        # Empirical Minimum
        min_idx = np.argmin(wnv)
        min_r_raw = rs[min_idx]

        # Convex Fit Minimum
        fit_x, fit_y = solve_convex_pwl(rs, wnv)
        min_idx_conv = np.argmin(fit_y)
        min_r_convex = fit_x[min_idx_conv]

        log(f"{label:<20} | {min_r_raw:.4f}       | {min_r_convex:.4f}")

    # Process Truncated
    sorted_levels = sorted(data_tr.keys())
    for lvl in sorted_levels:
        process_and_log(f"Truncated (L={lvl})", data_tr[lvl])

    # Process Untruncated
    if data_untr:
        process_and_log("Untruncated", data_untr)

    log(divider + "\n")


def fit_convex_pwl_curve(summary, data_dir, save_dir=None):
    """
    Plots the Computational Cost and Work-Normalized Variance with Convex Piecewise Linear fits.
    """
    if save_dir is None: save_dir = data_dir
    timestamp = summary["timestamp"]
    
    data_tr, data_untr = parse_results(summary)
    
    # Plot Settings
    COLORS = ["#F8DC94", "#DC647F", "#557858", "#564BAA"]
    color_untrunc = "#854672"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Helper to plot a single series
    def plot_series(ax, x, y, label, color, linestyle='-'):
        # Scatter raw points
        ax.scatter(x, y, alpha=0.4, color=color)
        # Solve and plot convex fit
        fit_x, fit_y = solve_convex_pwl(x, y)
        ax.plot(fit_x, fit_y, label=label, color=color, linestyle=linestyle, marker='.')

    # --- Plot Loops ---
    sorted_levels = sorted(data_tr.keys())
    
    # Subplot 1: Time vs Rate
    for i, lvl in enumerate(sorted_levels):
        subset = data_tr[lvl]
        plot_series(axes[0], [d["r"] for d in subset], [d["time"] for d in subset], 
                    label=rf'Truncated $M_t$={lvl}', color=COLORS[i % len(COLORS)])
        
    if data_untr:
        plot_series(axes[0], [d["r"] for d in data_untr], [d["time"] for d in data_untr],
                    label=rf'Untruncated $M_t=\infty$', color=color_untrunc, linestyle='--')

    # Subplot 2: WNV vs Rate
    for i, lvl in enumerate(sorted_levels):
        subset = data_tr[lvl]
        plot_series(axes[1], [d["r"] for d in subset], [d["work_norm_var"] for d in subset],
                    label=rf'Truncated $M_t$={lvl}', color=COLORS[i % len(COLORS)])

    if data_untr:
        plot_series(axes[1], [d["r"] for d in data_untr], [d["work_norm_var"] for d in data_untr],
                    label=rf'Untruncated $M_t=\infty$', color=color_untrunc, linestyle='--')

    # --- Formatting ---
    manual_xticks = np.sort(np.append(np.arange(0.51, 0.70001, 0.04) , 0.7))
    
    for i, ax in enumerate(axes):
        ax.set_xlabel(rf"Rate $r_t$")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.507, 0.703)
        ax.set_xticks(manual_xticks)
        ax.set_xticklabels([f"{t:.2f}" for t in manual_xticks])
        ax.legend()

    axes[0].set_ylabel("Time (s)")
    axes[1].set_ylabel("Work-Normalized Variance")
    
    # Scientific notation for Y-axis on the right plot
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0, 0))
    axes[1].yaxis.set_major_formatter(formatter)

    fname = os.path.join(save_dir, f"convex_pwl_{timestamp}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved Convex PWL Figure: {fname}")
    plt.close()

# ---------------------------------------------------------------------
#                  WORKERS FOR PARALLEL EXECUTION
# ---------------------------------------------------------------------
def run_untruncated_worker(seed_i, time_horizon, n_max, rate_untruncated, dimension):
    """
    One replicate of Untruncated MLMC
    """
    np.random.seed(seed_i)

    simulator = make_simulator(dimension)
    func, func_last = make_payoffs()
    funs = [func] * (time_horizon - 1) + [func_last] 
    rates = np.full(time_horizon - 1, rate_untruncated)

    res = np.zeros(n_max)
    for j in range(n_max):
        res[j] = Untruncated_MLMC(simulator, [], rates, funs, time_horizon)
    return res

def run_truncated_worker(seed_i, trunc_value, time_horizon, n_max, rate_truncated,
                          dimension):
    """
    One replicate of Truncated MLMC 
    """
    np.random.seed(seed_i)

    simulator = make_simulator(dimension)
    func, func_last = make_payoffs()
    funs = [func] * (time_horizon - 1) + [func_last]
    truncation_point = [trunc_value] * (time_horizon - 1)
    rates = np.full(time_horizon - 1, rate_truncated)

    res = np.zeros(n_max)
    for j in range(n_max):
        res[j] = Truncated_MLMC(simulator, [], truncation_point, rates, funs, time_horizon)
    return res

# ---------------------------------------------------------------------
#                            MAIN ENTRY
# ---------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Truncated MLMC / Untruncated MLMC Rate Search in Optimal Stopping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_horizon", type=int, default=3, help="T")
    parser.add_argument("--n_max", type=int, default=20000, help="number ofestimators per replicate")
    parser.add_argument("--n_rep", type=int, default=50, help="number of independent replicates")
    parser.add_argument("--dimension", type=int, default=1, help="m (state dimension)")
    
    # MLMC settings
    parser.add_argument("--trunc", type=int, default=[9,10,11], nargs='+', help="Truncation levels")
    parser.add_argument("--rate_truncated", type=float, default=np.round(np.arange(0.51, 0.71, 0.01), 2).tolist(),  help="Rates for truncated MLMC")
    parser.add_argument("--rate_untruncated", type=float, default=np.round(np.arange(0.51, 0.71, 0.01), 2).tolist(), help="Rates for untruncated MLMC")
    
    # loading only
    parser.add_argument("--load_json", type=str, default=None, help="Load JSON and replot instead of running")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--results_dir", type=str, default=SCRIPT_DIR)

    return parser.parse_args()

def load_from_json(json_path):
    json_dir = os.path.dirname(json_path)
    save_dir = os.path.join(json_dir, "rerun")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading experiment summary from: {json_path}")
    with open(json_path, "r") as f:
        summary = json.load(f)

    for trunc, info in summary.get("trunc_mlmc", {}).items():
        print(f"Truncated MLMC (trunc={trunc}): mean = {info['mean']:.6f}, std = {info['std']:.6f}")

    output_txt = os.path.join(save_dir, "optimal_rates.txt")
    # clear file
    open(output_txt, 'w').close()

    fit_convex_pwl_curve(summary, data_dir=json_dir, save_dir=save_dir) 
    print_optimal_rates(summary, output_file=output_txt)

if __name__ == "__main__":
    args = parse_arguments()

    if args.load_json is not None:
        load_from_json(args.load_json)
        exit(0)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(args.results_dir, f"rate_{timestamp}")
    os.makedirs(results_root, exist_ok=True)

    print(f"\nResults will be saved under: {results_root}")
    
    # Shared Params
    T = args.time_horizon
    n_max = args.n_max
    n_rep = args.n_rep
    
    # --------------------------
    #  Untruncated MLMC 
    # --------------------------
    untr_results = {}
    print("\n=== Running Untruncated MLMC ===")
    for r_untr in args.rate_untruncated:
        print(f"\n--- Rate r = {r_untr} ---")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_rep) as executor:
            futures = []
            for r_idx in range(n_rep):
                seed_i = args.seed + r_idx
                futures.append(
                    executor.submit(
                        run_untruncated_worker,
                        seed_i, T, n_max, r_untr, args.dimension,
                        
                    )
                )

            untr_arrays = [f.result() for f in futures]

        untr_results_array = np.vstack(untr_arrays)
        time_untr = time.time() - t0

        mean_untr = float(np.mean(untr_results_array))
        std_untr = float(np.std(untr_results_array) / np.sqrt(n_max * n_rep))
        
        r_vec_untr = np.full(T - 1, r_untr)
        cost_untr = cost_untruncated(r_vec_untr, T)

        file_name = f"untruncated_mlmc_rate{r_untr}_{timestamp}.npy"
        np.save(os.path.join(results_root, file_name), untr_results_array)

        untr_results[f"rate{r_untr:.3f}"] = {
            "rate": r_untr,
            "mean": mean_untr,
            "std":  std_untr,
            "file": file_name,
            "cost": cost_untr,
            "time": time_untr
        }
        print(f"Untruncated MLMC: Rate={r_untr:.3f} | mean = {mean_untr:.4f}, std = {std_untr:.6f},\
               time = {time_untr:.2f}, cost ≈ {cost_untr:.4f}")

    # -----------------------------
    #  Truncated MLMC 
    # -----------------------------
    tr_results = {}
    print("\n=== Running Truncated MLMC ===")
    for t_idx, trunc_value in enumerate(args.trunc):
        for r_tr in args.rate_truncated:
            print(f"\n--- Truncation level {trunc_value}, r_t = {r_tr} ---")
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=n_rep) as executor:
                futures = []
                for r_idx in range(n_rep):
                    seed_i = args.seed + 100 * t_idx + r_idx
                    futures.append(
                        executor.submit(
                            run_truncated_worker,
                            seed_i, trunc_value, T, n_max, r_tr,
                            args.dimension
                        )
                    )

                tr_arrays = [f.result() for f in futures]

            tr_results_array = np.vstack(tr_arrays)  # shape (n_rep, n_max)
            time_tr = time.time() - t0

            mean_trunc = float(np.mean(tr_results_array))
            std_trunc = float(np.std(tr_results_array) / np.sqrt(n_max * n_rep))

            trunc_pt = [trunc_value] * (T - 1)
            r_vec_tr = np.full(T - 1, r_tr)
            cost_trunc = cost_truncated(trunc_pt, r_vec_tr, T)

            file_name = f"truncated_mlmc_trunc{trunc_value}_rate{r_tr}_{timestamp}.npy"
            np.save(os.path.join(results_root, file_name), tr_results_array)

            tr_results[f"{trunc_value}_{r_tr:.3f}"] = {
                "trunc": trunc_value,
                "rate": r_tr,
                "mean": mean_trunc,
                "std":  std_trunc,
                "file": file_name,
                "cost": cost_trunc,
                "time": time_tr
            }
            print(f"Truncated MLMC: Trunc={trunc_value} Rate={r_tr:.3f} | \
                    mean = {mean_trunc:.4f}, std = {std_trunc:.6f}, \
                    time = {time_tr:.2f}, cost ≈ {cost_trunc:.4f}") 

    # -----------------------------
    # Save & Plot
    # -----------------------------
    summary = {
        "timestamp": timestamp,
        "global_params": vars(args),
        "trunc_mlmc": tr_results,
        "untrunc_mlmc": untr_results,
    }

    json_file = os.path.join(results_root, f"results_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved JSON: {json_file}")

    fit_convex_pwl_curve(summary, data_dir=results_root)
    
    output_txt = os.path.join(results_root, "optimal_rates.txt")
    print_optimal_rates(summary, output_file=output_txt)