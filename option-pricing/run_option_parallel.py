import numpy as np
import json, os, time, argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from option_setup import make_simulator, make_payoffs
from estimators import *

# ---------------------------------------------------------------------
#          AUXILIARY FUNCTIONS
# ---------------------------------------------------------------------
def calc_confidence_interval(data):
    """
    Calculates the 95% confidence interval for the mean of the input data.
    Returns tuple: (lower_bound, upper_bound)
    """
    # Flatten data to treat all replicates/estimators as a single sample set
    flat_data = data.flatten()
    n = flat_data.size
    
    mean = np.mean(flat_data)
    std_dev = np.std(flat_data)
    std_err = std_dev / np.sqrt(n)
    
    # Z-score for 95% confidence 
    z_score = 1.96
    
    margin_of_error = z_score * std_err
    return mean - margin_of_error, mean + margin_of_error
# ---------------------------------------------------------------------
#              ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Truncated MLMC / Untruncated MLMC Option Pricing")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_horizon", type=int, default=4, help="T")
    parser.add_argument("--n_max", type=int, default=10000, help="number of estimators per replicate")
    parser.add_argument("--n_rep", type=int, default=500, help="number of independent replicates")

    parser.add_argument("--volatility", type=float, default=0.2, help="sigma")
    parser.add_argument("--strike_price", type=float, default=100.0, help="K")
    parser.add_argument("--discount", type=float, default=0.05, help="gamma")

    parser.add_argument("--dimension", type=int, default=5, help="m (state dimension)")
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--delta", type=float, default=0.0, help="dividend yield")

    # MLMC settings
    parser.add_argument("--trunc", type=int, default=[9, 10, 11],
                        help="List of truncation levels to")

    parser.add_argument("--rate_truncated", type=float, default=[0.59, 0.58, 0.59], 
                        help="Rate parameter(s) for truncated MLMC (geometric parameter r)")
    
    parser.add_argument("--rate_untruncated", type=float, default=0.6,
                        help="Rate parameter for untruncated MLMC")
    # loading only
    parser.add_argument("--load_json", type=str, default=None,
                        help="If path is provided, load JSON and replot results instead of running.")
   
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--results_dir", type=str, default=os.path.join(SCRIPT_DIR, "results_option"))

    return parser.parse_args()

# ---------------------------------------------------------------------
#                    LOAD EXISTING RESULTS FROM JSON
# ---------------------------------------------------------------------
def load_from_json(json_path):
    json_dir = os.path.dirname(json_path)
    save_dir = os.path.join(json_dir, "rerun")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading experiment summary from: {json_path}")
    with open(json_path, "r") as f:
        summary = json.load(f)

    # Print stats
    untr = summary["untrunc_mlmc"]
    untr_ci_lo, untr_ci_hi = untr["ci_95"]
    print(f"\nUntruncated MLMC: mean = {untr['mean']:.6f}, std = {untr['std']:.6f}"
              f"CI = [{untr_ci_lo:.6f}, {untr_ci_hi:.6f}]")

    for key, tr in summary["trunc_mlmc"].items():
        tr_ci_lo, tr_ci_hi = tr["ci_95"]
        trunc = tr["trunc"]
        rate = tr["rate"]
        print(f"\nTruncated MLMC trunc={trunc}, rate={rate} | mean = {tr['mean']:.6f}, std = {tr['std']:.6f} "
                f"CI = [{tr_ci_lo:.6f}, {tr_ci_hi:.6f}]")


# ---------------------------------------------------------------------
#                  WORKERS FOR PARALLEL EXECUTION
# ---------------------------------------------------------------------
def run_untruncated_worker(seed_i, time_horizon, n_max, rate_untruncated,
                           discount, delta, sigma, dimension, S0, strike):
    """
    One replicate of Untruncated MLMC
    """
    np.random.seed(seed_i)

    simulator = make_simulator(discount, delta, sigma, dimension, S0)
    func, func_last = make_payoffs(strike, discount)
    funs = [func] * (time_horizon - 1) + [func_last] 
    
    rates = np.full(time_horizon - 1, rate_untruncated)

    res = np.zeros(n_max)
    for j in range(n_max):
        res[j] = Untruncated_MLMC(simulator, [], rates, funs, time_horizon)
    return res


def run_truncated_worker(seed_i, trunc_value, time_horizon, n_max, rate_truncated,
                         discount, delta, sigma, dimension, S0, strike):
    """
    One replicate of Truncated MLMC
    """
    np.random.seed(seed_i)

    simulator = make_simulator(discount, delta, sigma, dimension, S0)
    func, func_last = make_payoffs(strike, discount)
    funs = [func] * (time_horizon - 1) + [func_last]

    truncation_point = [trunc_value] * (time_horizon - 1)
    rates = np.full(time_horizon - 1, rate_truncated)

    res = np.zeros(n_max)
    for j in range(n_max):
        res[j] =  Truncated_MLMC(simulator, [], truncation_point, rates, funs, time_horizon)
    return res

# ---------------------------------------------------------------------
#                            MAIN ENTRY
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()

    # CASE 1: Only reload & replot
    if args.load_json is not None:
        load_from_json(args.load_json)
        exit(0)

    # Create results directory for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(args.results_dir, f"res_{timestamp}")
    os.makedirs(results_root, exist_ok=True)
    args.results_dir = results_root

    print(f"\nResults will be saved under: {results_root}")

    T = args.time_horizon
    n_max = args.n_max
    n_rep = args.n_rep

    # --------------------------
    #  Untruncated MLMC 
    # --------------------------
    print(f"\n=== Running Untruncated MLMC with rate = {args.rate_untruncated} ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_rep) as executor:
        futures = []
        for i in range(n_rep):
            seed_i = args.seed + 10000 + i  
            futures.append(
                executor.submit(
                    run_untruncated_worker,
                    seed_i, T, n_max, args.rate_untruncated,
                    args.discount, args.delta, args.volatility,
                    args.dimension, args.S0, args.strike_price
                )
            )

        untr_arrays = [f.result() for f in futures]

    untr_results_array = np.vstack(untr_arrays)  # shape (n_rep, n_max)
    time_untr = time.time() - t0

    mean_untrunc = float(np.mean(untr_results_array))
    std_untr = float(np.std(untr_results_array) / np.sqrt(n_max*n_rep))

    untr_ci_lo, untr_ci_hi = calc_confidence_interval(untr_results_array)

    file_untr = f"untruncated_mlmc_{timestamp}.npy"
    np.save(os.path.join(results_root, file_untr), untr_results_array)
    r_vec_untr = np.full(T - 1, args.rate_untruncated)
    cost_untr = cost_untruncated(r_vec_untr, T)
    untr_result = {
        "rate": args.rate_untruncated,
        "mean": mean_untrunc,
        "std": std_untr,
        "ci_95": [untr_ci_lo, untr_ci_hi],
        "file": file_untr,
        "cost": cost_untr,
        "time": time_untr
    }
    print(f"Untruncated MLMC: mean = {mean_untrunc:.6f}, std = {std_untr:.6f}, time = {time_untr:.2f}s, \
          cost = {cost_untr:.4f}, CI = [{untr_ci_lo:.6f}, {untr_ci_hi:.6f}]")

    # -----------------------------
    #  Truncated MLMC 
    # -----------------------------
    tr_results = {}

    print("\n=== Running Truncated MLMC ===")
    for i, (trunc_value, r_tr) in enumerate(zip(args.trunc, args.rate_truncated)):
            print(f"\n--- Truncation level {trunc_value}, r_t = {r_tr} ---")
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=n_rep) as executor:
                futures = []
                for r_idx in range(n_rep):
                    seed_i = args.seed + r_idx
                    futures.append(
                        executor.submit(
                            run_truncated_worker,
                            seed_i, trunc_value, T, n_max, r_tr,
                            args.discount, args.delta, args.volatility,
                            args.dimension, args.S0, args.strike_price
                        )
                    )

                tr_arrays = [f.result() for f in futures]

            tr_results_array = np.vstack(tr_arrays)  # shape (n_rep, n_max)
            time_tr = time.time() - t0

            mean_trunc = float(np.mean(tr_results_array))
            std_trunc = float(np.std(tr_results_array) / np.sqrt(n_max*n_rep))

            tr_ci_lo, tr_ci_hi = calc_confidence_interval(tr_results_array)

            truncation_point = [trunc_value] * (T - 1)
            r_vec_tr = np.full(T - 1, r_tr)
            cost_trunc = cost_truncated(truncation_point, r_vec_tr, T)

            file_trunc = f"truncated_mlmc_trunc{trunc_value}_rate{r_tr}_{timestamp}.npy"
            np.save(os.path.join(results_root, file_trunc), tr_results_array)

            tr_results[f"{trunc_value}_{r_tr:.3f}"] = {
                "trunc": trunc_value,
                "rate": r_tr,
                "mean": mean_trunc,
                "std":  std_trunc,
                "ci_95": [tr_ci_lo, tr_ci_hi],
                "file": file_trunc,
                "cost": cost_trunc,
                "time": time_tr
            }

            print(f"Truncated MLMC: Trunc={trunc_value} Rate={r_tr}: \
                   mean = {mean_trunc:.6f}, std = {std_trunc:.6f}, time = {time_tr:.2f}s \
                   cost ≈ {cost_trunc:.4f}, CI = [{tr_ci_lo:.6f}, {tr_ci_hi:.6f}]")

    # -----------------------------
    #  Save the results
    # -----------------------------
    summary = {
        "timestamp": timestamp,
        "global_params": vars(args),
        "untrunc_mlmc": untr_result,
        "trunc_mlmc": tr_results
    }

    json_file = os.path.join(results_root, f"results_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved JSON: {json_file}")

