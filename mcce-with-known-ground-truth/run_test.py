import numpy as np
import os, argparse
from concurrent.futures import ProcessPoolExecutor
from estimators import *
from visualization import plot_results
from problem_setup import simulator_gaussian_martingale, funs
import time
from datetime import datetime
# from tqdm import tqdm
import json

# ---------------------------------------------------------------------
#              ARGUMENT PARSING
# ---------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Truncated MLMC / Untruncated MLMC / SAA Test Example")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_horizon", type=int, default=3, help="T")
    parser.add_argument("--n_max", type=int, default=100000, help="number of trees per replicate")
    parser.add_argument("--n_rep", type=int, default=10, help="number of independent replicates")
    parser.add_argument("--true_value", type=float, default=float(np.exp(-0.5)), 
                        help="True value of the nested expectation F(x)")
    # MLMC settings
    parser.add_argument("--trunc", type=int, default=[6, 5],
                        help="List of truncation levels M1, M2")
    parser.add_argument("--rate_truncated", nargs="+", type=float, default=[1-2**(-3/2), 1-2**(-5/4)],
                        help="Rate parameters for truncated MLMC (geometric parameter r)")
    parser.add_argument("--rate_untruncated", type=float, default=[0.74, 0.6],
                        help="Rate parameters for untruncated MLMC")
    # SAA
    parser.add_argument("--sample_size", nargs="+", type=str, default="[[3600, 60, 60], [200, 200, 200]]",
                        help="Sample size for SAA (n_1, n_2, ... , n_T)")

    # loading only
    parser.add_argument("--load_json", type=str, default=None,
                        help="If path is provided, load JSON and replot results instead of running.")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--results_dir", type=str, default=os.path.join(SCRIPT_DIR, "results"))
    return parser.parse_args()
def parse_saa_configs(raw: str):
    """
    Parse --sample_size string into a list of configurations.
    Expected format: JSON-like
    """
    cfg = json.loads(raw)
    if isinstance(cfg, list) and len(cfg) > 0 and isinstance(cfg[0], int):
        # Single config [200,200,200] -> [[200,200,200]]
        return [cfg]
    if not (isinstance(cfg, list) and all(isinstance(x, list) for x in cfg)):
        raise ValueError("Sample_size must be a JSON list of lists, e.g. '[[200,200,200],[3600,60,60]]' or '[3600,60,60]'.")
    return cfg

def load_from_json(json_path):
    json_dir = os.path.dirname(json_path)
    save_dir = os.path.join(json_dir, "rerun")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading experiment summary from: {json_path}")
    with open(json_path, "r") as f:
        summary = json.load(f)

    # Untruncated 
    untr = summary["untrunc_mlmc"]
    print(f"\nUntruncated MLMC: mean = {untr['mean']:.6f}, \
          std = {untr['std']:.6f}, cost = {untr['cost']:.4f}")
    # Truncated 
    tr = summary["trunc_mlmc"]
    print(f"\nTruncated MLMC: mean = {tr['mean']:.6f}, \
          std = {tr['std']:.6f}, cost = {tr['cost']:.4f}")

    # SAA (multiple configs)
    for key, info in summary["saa"].items():
        print(
            f"SAA (n={info['sample_size']}): mean = {info['mean']:.6f}, "
            f"std = {info['std']:.6f}, cost = {info['cost']:.4f}"
        )
    plot_results(summary, data_dir=json_dir, save_dir=save_dir)

def run_untruncated_worker(seed_i, time_horizon, n_max, rate_untruncated, funs, simulator):
    """
    One replicate of Untruncated MLMC
    """
    np.random.seed(seed_i)
    res = np.zeros(n_max)
    for j in range(n_max):
        res[j] = Untruncated_MLMC(simulator, [], rate_untruncated, funs, time_horizon)
    return res    
  
def run_truncated_worker(seed_i, trunc_value, time_horizon, n_max, rate_truncated, funs, simulator):
    """
    One replicate of Truncated MLMC for a given truncation level.
    """
    np.random.seed(seed_i)
    res = np.zeros(n_max)
    for j in range(n_max):
        res[j] = Truncated_MLMC(simulator, [], trunc_value, rate_truncated, funs, time_horizon)
    return res

def run_saa_worker(seed_i, time_horizon, sample_sizes, simulator):
    """
    One replicate of SAA for the given sample sizes.
    """
    np.random.seed(seed_i)
    n1 = sample_sizes[0]
    inner_sizes = sample_sizes[1:]

    res = np.zeros(n1)
    for j in range(n1):
        res[j] = SAA(
            simulator,
            [],
            inner_sizes,
            funs,
            time_horizon=time_horizon,
        )
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
    true_val = float(args.true_value)
    # --------------------------
    #  Untruncated MLMC 
    # --------------------------
    print(f"\n=== Running Untruncated MLMC with rate = {args.rate_untruncated} ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_rep) as executor:
        futures = []
        for i in range(n_rep):
            seed_i = args.seed  + i   # separate stream for RU
            futures.append(
                executor.submit(
                    run_untruncated_worker,
                    seed_i, T, n_max, args.rate_untruncated,
                    funs, simulator_gaussian_martingale
                )
            )

        untr_arrays = [f.result() for f in futures]

    untr_results_array = np.vstack(untr_arrays)  # shape (n_rep, n_max)
    time_ru = time.time() - t0

    mean_untrunc = float(np.mean(untr_results_array))
    std_untr = float(np.std(untr_results_array) / np.sqrt(n_max*n_rep))
    mse_untr = np.mean( (untr_results_array - true_val)**2 )

    file_untr = f"untrunc_mlmc_{timestamp}.npy"
    np.save(os.path.join(results_root, file_untr), untr_results_array)
    cost_untr = cost_untruncated(args.rate_untruncated, T)
    untr_result = {
        "rate": args.rate_untruncated,
        "mean": mean_untrunc,
        "std" :  std_untr,
        "file": file_untr,
        "cost": cost_untr,
        "time": time_ru
    }

    print(f"Untruncated MLMC: mean = {mean_untrunc:.6f}, std = {std_untr:.6f}, time = {time_ru:.2f}s, cost = {cost_untr:.4f}")
    # -------------------------
    #  Truncated MLMC 
    # -------------------------
    # to write it with 4 significant digits
    rate_list = args.rate_truncated if isinstance(args.rate_truncated, (list, tuple)) else [args.rate_truncated]
    rate_tag = "-".join(f"{r:.4g}" for r in rate_list)
    print(f"\n=== Running Truncated MLMC with rate = {rate_tag} and truncation ={args.trunc} ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_rep) as executor:
        futures = []
        for i in range(n_rep):
            seed_i = args.seed + i
            futures.append(
                executor.submit(
                    run_truncated_worker,
                    seed_i, args.trunc, T, n_max, args.rate_truncated, funs,
                    simulator_gaussian_martingale
                )
            )
        rep_arrays = [f.result() for f in futures]

    res_tr = np.vstack(rep_arrays)  # shape (n_rep, n_max)
    time_tr = time.time() - t0

    mean_trunc = float(np.mean(res_tr))
    std_trunc = float(np.std(res_tr) / np.sqrt(n_max*n_rep))
    mse_tr = np.mean( (res_tr - true_val)**2 )
    cost_trunc = cost_truncated(args.trunc, args.rate_truncated, T)

    file_trunc = f"trunc_mlmc_trunc{args.trunc}_rate{rate_tag}_{timestamp}.npy"
    np.save(os.path.join(results_root, file_trunc), res_tr)

    tr_result = {
        "trunc": args.trunc,
        "rate": args.rate_truncated,
        "mean": mean_trunc,
        "std":  std_trunc,
        "file": file_trunc,
        "cost": cost_trunc,
        "time": time_tr
    }
    print(f"Truncated MLMC: mean = {mean_trunc:.6f}, std = {std_trunc:.6f}, "
        f"time = {time_tr:.2f}s, cost ≈ {cost_trunc:.4f}")
    
    # -----------------------------
    #  SAA 
    # -----------------------------
    raw_saa_configs = parse_saa_configs(args.sample_size)
    saa_configs = []

    # Generate SAA configurations    
    for cfg in raw_saa_configs:
        n1, n2, n3 = cfg[0], cfg[1], cfg[2]
        print(f"n1:{n1}, n2:{n2}, n3:{n3}")
        # Config 1 (n1 = n2 = n3)
        if n1 == n2 == n3:
            if n1 <= 80:
                steps = list(range(2, n1 + 1, 2))
            elif n1 > 80 and n1 <= 201:
                steps = list(range(2, n1 + 1, 8)) + [n1]
            else:
                steps = list(range(10, n1 + 1, 15))
            # print(f"  -> SAA (Max n={n1}). Generating {len(steps)} steps: {steps}")
            for s in steps:
                saa_configs.append([s, s, s])
        # Config 2: (n1 = n2^2 = n3^2)
        elif n1 == n2**2 and n2 == n3:
            if n2 <= 80:
                steps = list(range(2, n2 + 1, 2))
            else:
                steps = list(range(4, n2 + 1, 4))
            # print(f"  -> SAA (Max n={n2}). Generating {len(steps)} steps: {steps}")
            for s in steps:
                saa_configs.append([s**2, s, s])
        else:
            print(f"  -> Warning: Config {cfg} does not match standard SAA patterns.")
    

    saa_results = {}
    print(f"\n=== Running SAA ({len(saa_configs)} configurations) ===")    
    seed_idx = args.seed
    for idx, cfg in enumerate(saa_configs):
        cfg_key = str(cfg)
        print(f"\n--- SAA config {cfg_key} ---")
        t0 = time.time()
        seed_idx = seed_idx + idx * 1000
        with ProcessPoolExecutor(max_workers=n_rep) as executor:
            futures = []
            for i in range(n_rep):
                seed_i = seed_idx + i  
                futures.append(
                    executor.submit(
                        run_saa_worker,
                        seed_i,
                        T,
                        cfg,
                        simulator_gaussian_martingale
                    )
                )
            saa_arrays = [f.result() for f in futures]
        saa_results_array = np.vstack(saa_arrays)  # shape (n_rep, n1)
        time_saa = time.time() - t0
        saa_estimators = np.mean(saa_results_array, axis=1) # Shape (n_rep,)
        saa_mse = np.mean( (saa_estimators - true_val)**2 )
        saa_std = float(np.std(saa_estimators) / np.sqrt(n_rep)) 
        saa_mean = float(np.mean(saa_results_array))

        saa_file = f"saa_{idx}_{cfg_key}_{timestamp}.npy"
        np.save(os.path.join(results_root, saa_file), saa_results_array)

        cost_saa_val = cost_saa(cfg, T)

        saa_results[cfg_key] = {
            "sample_size": cfg,
            "mean": saa_mean,
            "std" : saa_std,
            "file": saa_file,
            "cost": cost_saa_val,
            "time": time_saa
        }

        print(
            f"SAA (n={cfg}): mean = {saa_mean:.6f}, std = {saa_std:.6f}, "
            f"time = {time_saa:.2f}s"
        )
    # -----------------------------
    # Save consolidated JSON
    # -----------------------------
    summary = {
        "timestamp": timestamp,
        "global_params": {
            **vars(args),
            "true_value": true_val,
        },
        "untrunc_mlmc": untr_result,
        "trunc_mlmc": tr_result,
        "saa": saa_results,
    }

    json_file = os.path.join(results_root, f"results_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)

    # -------------------------------------------------
    #  Plot and collect slopes
    # -------------------------------------------------
    slopes = plot_results(summary, data_dir=results_root)

    # attach slopes and overwrite JSON
    summary["slopes"] = slopes
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Updated JSON with slopes: {json_file}")
