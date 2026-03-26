# run_ctx.py
import os, json, argparse
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import visualization as viz

from empirical_solver import solve_exact_robust_problem
from parallel_workers import init_parallel_worker, run_single_rep_worker
from helpers import (
    train_adam,
    build_env, 
)
from estimators import make_estimator_saa, make_estimator_mlmc
from utils import (
    _extract_run_timestamp,
    _lr_to_display,
    _lr_to_filename_tag,
    _normalize_lr_cfg,
    _parse,
    parse_mlmc_config,
    set_seed,
)

# --- MAIN EXECUTION ---

def main():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--data_path", default="dataset.xlsx")
    p.add_argument("--outdir", default="results/")
    p.add_argument("--load_existing", default=None, help="Path to the json file")

    
    # Run Params
    p.add_argument("--K", default=2000, type=int, help="number of SGD iterations")
    p.add_argument("--num_runs", default=20, type=int, help="number of simulations")
    p.add_argument("--num_workers", default=4, type=int, help="Parallel workers over num_runs; 1 disables parallelism")
    p.add_argument("--x_initial", default="[50, 0.0, 0.0]", help="Initial dual variable, and policy parameters (lambda, theta_1, theta_2)")
    p.add_argument("--seed", default=1234, type=int)
    
    # Grids
    p.add_argument("--saa_sizes", default="[[100, 200, 200], [100, 300, 300], [100, 500, 500], [100, 1000, 1000]]", help="[n_1,n_2,n_3] (sample sizes)")
    p.add_argument("--mlmc_sizes", default="[ [1000,[9,5],[0.5,0.5]], [1000,[10,6],[0.5,0.5]]]", help="[n_1,M,r] (M:truncation | r:rate parameter | n_1:outer batch size)")
    p.add_argument(
        "--learning_rates",
        default="[[0.75, 0.025]]", 
        help="ADAM learning rates: scalar (e.g. [0.75]) or split pair [lr_lambda, lr_theta] (e.g. [[0.75,0.25]])", 
    )
    # Clipping
    p.add_argument("--clip_lambda", default=100.0, type=float, help="Maximum absolute value for the lambda gradient. If 0.0, no clipping is applied")
    p.add_argument("--clip_thetas", default=[50.0,50.0], type=float, help="Maximum L2 norm for the theta gradients. If 0.0, no clipping is applied")
    
    # Environment
    p.add_argument("--r_c", default=0.4, type=float, help="Radius for context distribution shift")
    p.add_argument("--r_y", default=0.15, type=float, help="Radius for cost distribution shift")
    p.add_argument("--mu", default=2.0, type=float, help="Softmax (inverse) temperature (bigger = closer to max)")
    p.add_argument("--cost_params", default="[[3, 5, 5.5, 1], [1.7, 3.5, 3, 1]]", help="Mean costs for the cost_ymean function")
    p.add_argument("--covariance", default="[[5, 2.5], [2.5, 5]]", help="covariance matrix for actions:row 0->action0 | row1->action1")
    p.add_argument("--shift", default="[0.1, 0.1]", help="Cost shift on the means of a0 and a1")
    p.add_argument("--gamma_1", default=0.005, type=float, help="Quadratic penalty on theta1: gamma_1 * theta1^2")
    p.add_argument("--gamma_2", default=0.005, type=float, help="Quadratic penalty on theta2: gamma_2 * theta2^2")
    
    p.add_argument("--run_saa", default="True", help="True if an SAA instance is being run")
    p.add_argument("--run_mlmc", default="True", help="True if an MLMC instance is being run")

    args = p.parse_args()
    args.num_workers = max(1, int(args.num_workers))

    # Logic: Load or Run
    if args.load_existing:
        print(f"Loading: {args.load_existing}")
        json_path = args.load_existing
        base_dir = os.path.dirname(json_path)
        replots_dir = os.path.join(base_dir, "replots")
        os.makedirs(replots_dir, exist_ok=True)

        with open(json_path, "r") as f:
            loaded = json.load(f)

        # --- Rebuild args from JSON meta (so grids and clipping match) ---
        meta = loaded.get("meta", {})
        if not meta:
            raise ValueError(f"JSON file {json_path} missing 'meta' field.")

        # Recreate args Namespace purely from the JSON meta
        # Keep load_existing pointing to the JSON you loaded
        args = argparse.Namespace(**{**meta, "load_existing": json_path})

        # --- Load env_args if present; otherwise reconstruct from args ---
        env_args = loaded.get("env_args", None)
        if env_args is None:
            env_args = {
                "data_path": args.data_path,
                "r_c": args.r_c,
                "r_y": args.r_y,
                "mu": args.mu,
                "covariance": _parse(args.covariance),
                "cost_params": _parse(args.cost_params),
                "shift": _parse(args.shift),
                "gamma_1": float(getattr(args, "gamma_1", 0.0)),
                "gamma_2": float(getattr(args, "gamma_2", 0.0)),
            }
        else:
            env_args.setdefault("gamma_1", float(getattr(args, "gamma_1", 0.0)))
            env_args.setdefault("gamma_2", float(getattr(args, "gamma_2", 0.0)))

        # --- Load exact solution info from JSON 
        if "true_opt" not in loaded:
            raise ValueError(f"JSON file {json_path} missing 'true_opt' field.")
        exact = {"x_opt": np.array(loaded["true_opt"], dtype=float)}
        if "true_min" in loaded:
            exact["fun"] = float(loaded["true_min"])

        meta_recs = loaded.get("runs", [])

        # --- TEMP: HARDCODED FILTERING ------ DELETE BELOW LATER
        print(f"Records before filtering: {len(meta_recs)}")
        kept_recs = []
        for r in meta_recs:
            fname = r['file']
            
            # Skip.....
            if "lrlam1_th0.025" in fname:
                continue
            if "lrlam1_th0.01" in fname:
                continue
            if "lrlam0.75_th0.01" in fname:
                continue
            if "SAA" in fname and "n1_" in fname:
                continue
                
            kept_recs.append(r)
            
        meta_recs = kept_recs
        print(f"Records after filtering: {len(meta_recs)}")
        # --------------------------------- delete this (above) later ---------

        print(f"Loaded True Opt: {exact['x_opt']}")
        print(f"Loaded {len(meta_recs)} run records.")

        process_and_plot(replots_dir, meta_recs, args, exact, base_dir)
        return

    # Setup Environment
    env_args = {
        'data_path': args.data_path, 'r_c': args.r_c, 'r_y': args.r_y, 'mu': args.mu,
        'covariance': _parse(args.covariance), 
        'cost_params': _parse(args.cost_params), 
        'shift': _parse(args.shift),
        'gamma_1': args.gamma_1,
        'gamma_2': args.gamma_2,
    }
    torch.set_default_dtype(torch.float64)
    SIM_ENV = build_env(**env_args)
    print("Computing exact solution...")
    exact = solve_exact_robust_problem(SIM_ENV, _parse(args.x_initial))
    print(f"True Opt: {exact['x_opt']}")
    # exit()

    # Create a timestamped directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"res_{ts}")
    files_dir = os.path.join(outdir, "files")
    os.makedirs(files_dir, exist_ok=True)
    print(f"Results will be saved under {outdir}")

    # Initialize JSON
    json_path = os.path.join(outdir, f"res_{ts}.json")
    initial_data = {
        "meta": vars(args),
        "true_opt": exact["x_opt"].tolist(),
        "runs": []
    }
    with open(json_path, 'w') as f:
        json.dump(initial_data, f, indent=2)

    # Helper to append records incrementally
    def append_to_json(record):
        with open(json_path, 'r+') as f:
            data = json.load(f)
            data['runs'].append(record)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    # Parse boolean flags
    run_saa = (str(args.run_saa).lower() == "true")
    run_mlmc = (str(args.run_mlmc).lower() == "true")
    # --- Parse Simulation Grids ---
    saa_grid = _parse(args.saa_sizes) if run_saa else []
    if saa_grid and not isinstance(saa_grid[0], list): saa_grid = [saa_grid]
    
    mlmc_grid = _parse(args.mlmc_sizes) if run_mlmc else []
    if mlmc_grid and not isinstance(mlmc_grid[0], list): mlmc_grid = [mlmc_grid]
    mlmc_grid = [parse_mlmc_config(x) for x in mlmc_grid]
    # Parse Hyperparameters
    lrs = _parse(args.learning_rates)
    if not isinstance(lrs, list):
        lrs = [lrs]
    lrs = [_normalize_lr_cfg(lr) for lr in lrs]
    x_initial_vec = _parse(args.x_initial)
    x_t = torch.tensor(x_initial_vec, dtype=torch.get_default_dtype())
    clip_thetas = _parse(args.clip_thetas)
    
    all_recs = []
    pool = None
    if args.num_workers > 1:
        worker_state = {
            "env_args": env_args,
            "K": int(args.K),
            "x_initial": x_initial_vec,
            "clip_lambda": float(args.clip_lambda),
            "clip_thetas": clip_thetas,
        }
        print(f"Parallelizing num_runs with {args.num_workers} workers (single parent JSON writer).")
        pool = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_parallel_worker,
            initargs=(worker_state,),
        )

    def run_reps_for_lr(method, cfg_idx, size_cfg, lr, lr_idx):
        """
        Run args.num_runs repetitions for ONE (method, config_idx, lr).
        Returns a list of rec_dict entries to be plotted together.
        """
        recs = []

        # --- Build Gradient Estimator & label ---
        if method == "SAA":
            n1, nu, ny = size_cfg
            lbl = f"n{n1}_{nu}_{ny}"
            est_fn = None if pool is not None else make_estimator_saa(SIM_ENV, (nu, ny))
        else:
            n1, trunc, rate = size_cfg
            lbl = f"n{n1}_M({trunc[0]},{trunc[1]})_r({rate[0]},{rate[1]})"
            est_fn = None if pool is not None else make_estimator_mlmc(SIM_ENV, trunc, rate, args.clip_lambda, clip_thetas)

        jobs = []
        for r_idx in range(args.num_runs):
            seed_val = int(args.seed + (cfg_idx * 1e5) + (lr_idx * 1e3) + r_idx)
            lr_tag = _lr_to_filename_tag(lr)
            fname = f"{method}_idx{cfg_idx}_{lbl}_run{r_idx}_lr{lr_tag}.npy"
            jobs.append({
                "method": method,
                "config_idx": int(cfg_idx),
                "size_cfg": size_cfg,
                "lr": lr,
                "run_idx": int(r_idx),
                "seed_val": seed_val,
                "files_dir": files_dir,
                "fname": fname,
            })

        if pool is None:
            for job in jobs:
                r_idx = job["run_idx"]
                seed_val = job["seed_val"]
                fname = job["fname"]
                set_seed(seed_val)
                print(f"[{method}] {lbl} | LR:{_lr_to_display(lr)} | Run:{r_idx} | Seed:{seed_val}")
                res = train_adam(
                    x_t,
                    args.K,
                    n1,
                    lr,
                    est_fn,
                )
                res.update({
                    "config_idx": cfg_idx,
                    "run_idx": r_idx,
                    "step": [lr],
                    "method": method,
                })
                np.save(os.path.join(files_dir, fname), res)
                rec_dict = {"method": method, "file": fname, "config_idx": cfg_idx}
                append_to_json(rec_dict)
                recs.append(rec_dict)
            return recs

        print(f"[{method}] {lbl} | LR:{_lr_to_display(lr)} | Parallel runs: {args.num_runs}")
        futures = [pool.submit(run_single_rep_worker, job) for job in jobs]
        payloads = []
        for fut in as_completed(futures):
            payloads.append(fut.result())

        # Stable ordering in JSON: preserve run_idx order regardless of completion order.
        payloads.sort(key=lambda x: int(x["run_idx"]))
        for payload in payloads:
            rec_dict = payload["rec_dict"]
            append_to_json(rec_dict)
            recs.append(rec_dict)

        return recs
    
    # ---------------------------------------------------------
    #  Main Loops
    # ---------------------------------------------------------
    max_idx = max(len(saa_grid), len(mlmc_grid))
    try:
        for cfg_idx in range(max_idx):
            for lr_idx, lr in enumerate(lrs):
                # Run SAA
                if run_saa and cfg_idx < len(saa_grid):
                    all_recs.extend(run_reps_for_lr("SAA", cfg_idx, saa_grid[cfg_idx], lr, lr_idx))

                # Run MLMC
                if run_mlmc and cfg_idx < len(mlmc_grid):
                    all_recs.extend(run_reps_for_lr("MLMC", cfg_idx, mlmc_grid[cfg_idx], lr, lr_idx))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # --- PLOT ONCE AT THE END ---
    if all_recs:
        process_and_plot(outdir, all_recs, args, exact)

    print(f"Saved under {outdir}")
    
def process_and_plot(outdir, meta_recs, args, exact, data_dir=None):
    print("Plotting...")
    if data_dir is None:
        data_dir = outdir

    run_ts = _extract_run_timestamp(data_dir, outdir)
    combined_tag = run_ts
    all_cfg_dir = os.path.join(outdir, "plots", "plots_all_configs")
    os.makedirs(all_cfg_dir, exist_ok=True)

    grouped = defaultdict(list)
    for r in meta_recs: grouped[r['config_idx']].append(r)

    # --- RE-PARSE GRIDS for Filename Generation ---
    run_saa = (str(args.run_saa).lower() == "true")
    run_mlmc = (str(args.run_mlmc).lower() == "true")
    
    saa_grid = _parse(args.saa_sizes) if run_saa else []
    if saa_grid and not isinstance(saa_grid[0], list): saa_grid = [saa_grid]
    
    mlmc_grid = _parse(args.mlmc_sizes) if run_mlmc else []
    if mlmc_grid and not isinstance(mlmc_grid[0], list): mlmc_grid = [mlmc_grid]
    mlmc_grid = [parse_mlmc_config(x) for x in mlmc_grid]

    d_saa_global = []
    d_mlmc_global = []

    # --- MAIN DATA LOADING LOOP ---
    for idx, items in grouped.items():
        for x in items:
            path = os.path.join(data_dir, "files", x['file'])
            if os.path.exists(path):
                try:
                    data = np.load(path, allow_pickle=True).item()
                    
                    # # Filter Broken Runs
                    # if np.isnan(data['theta1']).any() or np.isnan(data['lambda_']).any():
                    #     print(f"Skipping broken run: {x['file']}")
                    #     continue

                    # Create a unique label for plotting
                    if x['method'] == "SAA":
                        if idx < len(saa_grid):
                            n1, n2, n3 = saa_grid[idx]
                            run_label = rf"SAA $(n_1, n_2, n_3) = ({n1}, {n2}, {n3})$"
                        else:
                            run_label = f"SAA_Cfg{idx}"
                    else: # MLMC
                        if idx < len(mlmc_grid):
                            n1, (m1, m2), (r1, r2) = mlmc_grid[idx]
                            run_label = rf"MLMC ($n_1 = {n1}, M = ({m1}, {m2}), r = ({r1}, {r2})$)"
                        else:
                            run_label = rf"MLMC_Cfg{idx}"

                    data["plot_label"] = run_label

                    # Add to lists
                    if x['method'] == "SAA": 
                        d_saa_global.append(data)
                    else: 
                        d_mlmc_global.append(data)

                except Exception as e:
                    print(f"Error loading {x['file']}: {e}")

    # --- Plotting ---
    if d_saa_global or d_mlmc_global:
        print(f"Generating plots for {len(d_saa_global)} SAA runs and {len(d_mlmc_global)} MLMC runs...")

        viz.plot_all_three(
            d_saa_global, d_mlmc_global,
            "",
            os.path.join(outdir, f"all_three_samples_{combined_tag}.pdf"),
            true_vals=exact["x_opt"],
        )
    print("Done.")

if __name__ == "__main__":
    main()
