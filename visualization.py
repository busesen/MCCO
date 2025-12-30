import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np
import statsmodels.api as sm

plt.rcParams.update({
    "text.usetex": True,  # Enable LaTeX rendering
    "font.family": "serif",
    "font.serif": ["Times"],  # Match LaTeX font
    "font.size": 22,         # Increase overall font size (including legend) - default=10
    "axes.labelsize": 22,  # Increase font size of axis labels (xlabel, ylabel) - default=12
    "xtick.labelsize": 20,  # Increase font size of x-axis tick labels (numbers) - default=10
    "ytick.labelsize": 20,  # Increase font size of y-axis tick labels (numbers) - default=10
    "xtick.major.size": 8,  # Length of major ticks - default=10
    "ytick.major.size": 8,  # - default=10
    "xtick.major.width": 1,  # Thickness of major ticks - default=2
    "ytick.major.width": 1,
    })
plt.rcParams["text.latex.preamble"] = r"""\usepackage{amsmath,amsfonts,amssymb} \usepackage{times}"""

# ---------------------------------------------------------------------
#              UTILITIES
# ---------------------------------------------------------------------
def cummean(x):
    """Calculate the cumulative mean of an array."""
    return np.cumsum(x) / np.arange(1, len(x) + 1)

def cumsd(x):
    """Calculate the cumulative standard deviation of an array."""
    cm_x = cummean(x)  # Cumulative mean of x
    cm_xx = cummean(x**2)  # Cumulative mean of x squared  
    csd_x = np.sqrt(cm_xx - cm_x**2)  # Cumulative standard deviation
    return csd_x

# ---------------------------------------------------------------------
#              PLOT
# ---------------------------------------------------------------------
def plot_results(summary, data_dir, save_dir=None):
    """
    Reproduce:
      1) log(MSE) vs log(total sampling cost)
      2) Running averages + 95% CIs

    using the consolidated JSON summary and the saved .npy arrays.
    Handles:
      - 1 Untruncated MLMC config
      - 1 Truncated MLMC config
      - possibly multiple SAA configs
    """
    if save_dir is None:
        save_dir = data_dir

    os.makedirs(save_dir, exist_ok=True)
    slopes = {
        "untrunc_mlmc": {},
        "trunc_mlmc": {},
        "saa": {} 
    }

    true_val = float(summary["global_params"]["true_value"])
    n_max = int(summary["global_params"]["n_max"])

    # Load Untruncated MLMC
    untrunc_info = summary["untrunc_mlmc"]
    untrunc_arr = np.load(os.path.join(data_dir, untrunc_info["file"]))
    cost_untrunc = float(untrunc_info["cost"])

    # Load Truncated MLMC 
    trunc_info = summary["trunc_mlmc"]
    trunc_arr = np.load(os.path.join(data_dir, trunc_info["file"]))
    cost_trunc = float(trunc_info["cost"])

    # Decide on n_plt
    exp_max = np.log10(n_max)
    if exp_max < 6:
        n_plt = np.unique(np.floor(10**np.arange(0.5, exp_max+0.00001, 0.15)).astype(int))
    else:
        n_plt = np.unique(np.floor(10**np.arange(0.5, exp_max+0.00001, 0.2)).astype(int))

    def mse_from_array(arr):
        temp = np.apply_along_axis(cummean, 1, arr)  # shape (n_rep, n_max)
        return np.mean((temp - true_val) ** 2, axis=0) # shape (n_max,)

    # MSEs
    MSE_untrunc = mse_from_array(untrunc_arr) 
    MSE_trunc = mse_from_array(trunc_arr)

    log_mse_untrunc = np.log10(MSE_untrunc[n_plt - 1])
    log_mse_trunc =  np.log10(MSE_trunc[n_plt - 1])

    x_untrunc = np.log10(n_plt * cost_untrunc)
    x_trunc = np.log10(n_plt * cost_trunc)

    X_untrunc = sm.add_constant(x_untrunc) # adds a constant (intercept) term to the data
    X_trunc = sm.add_constant(x_trunc)
    
    # Define range for plotting lines (e.g., Cost 10^1 to 10^7)
    x_line_range = np.linspace(1.0, 7, 100)

    model_untrunc = sm.OLS(log_mse_untrunc, X_untrunc).fit()
    line_untr = model_untrunc.params[0] + model_untrunc.params[1] * x_line_range
    slope_untrunc = model_untrunc.params[1]

    model_trunc = sm.OLS(log_mse_trunc, X_trunc).fit()
    line_tr = model_trunc.params[0] + model_trunc.params[1] * x_line_range
    slope_trunc =  model_trunc.params[1]

    slopes["untrunc_mlmc"]["mse"] = float(slope_untrunc)
    slopes["trunc_mlmc"]["mse"] = float(slope_trunc)

    print(f"Slope of Untruncated MLMC: {slope_untrunc:.6f}")
    print(f"Slope of Truncated MLMC: {slope_trunc:.6f}")

    # -----------------------------------------------------
    # SAA: Aggregate results from multiple config files
    # -----------------------------------------------------
    saa_infos = summary["saa"]
    saa_data = {
        "SAA(n1=n2=n3)": {"x": [], "y_mse": [], "color": "forestgreen", "marker": "x", "style": (0, (3, 1, 1, 1, 1, 1)), "label": "SAA1"},
        "SAA(n1=n2^2=n3^2)": {"x": [], "y_mse": [], "color": "#800080", "marker": "o", "style": "dashed", "label": "SAA2"}
    }
    for k, info in saa_infos.items():
        arr = np.load(os.path.join(data_dir, info["file"])) # shape (n_rep, n1)    
        ss = info["sample_size"]
        estimators = np.mean(arr, axis=1)
        MSE_saa = np.mean( (estimators - true_val)**2 ) 

        cost_saa = float(info["cost"])
        # Categorize into Green (SAA 1) or Purple (SAA 2)
        if ss[0] == ss[1] == ss[2]:
            grp = "SAA(n1=n2=n3)"
        else:
            grp = "SAA(n1=n2^2=n3^2)"
        
        saa_data[grp]["x"].append(np.log10(cost_saa))
        saa_data[grp]["y_mse"].append(np.log10(MSE_saa))

    # Fit regressions for SAA
    for grp in ["SAA(n1=n2=n3)", "SAA(n1=n2^2=n3^2)"]:
        if len(saa_data[grp]["x"]) > 1:
            X_saa = sm.add_constant(saa_data[grp]["x"])
            # Slope
            model_saa = sm.OLS(saa_data[grp]["y_mse"], X_saa).fit()
            saa_data[grp]["slope_mse"] = model_saa.params[1]
            saa_data[grp]["line_mse"] = model_saa.params[0] + model_saa.params[1] * x_line_range
            slopes["saa"][grp] = {"mse": float(model_saa.params[1])}

    # Running averages + 95% CIs (single replicate per method)
    num_call_exponent = int(round(np.log10(n_max / 5.0)))
    points = np.arange(5 * 10 ** (num_call_exponent - 2), n_max + 1, 10 ** (num_call_exponent - 3)).astype(int)
    def ci_curves(arr):
        y = arr[0, :]
        est = cummean(y)
        sd = cumsd(y)
        upper = est[points - 1] + 1.96 * sd[points - 1] / np.sqrt(points)
        lower = est[points - 1] - 1.96 * sd[points - 1] / np.sqrt(points)
        return est[points - 1], lower, upper

    est_untr, lower_untr, upper_untr = ci_curves(untrunc_arr)
    est_tr,   lower_tr,   upper_tr   = ci_curves(trunc_arr)

    def new_fig_ax(figsize=(10,6)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.12, 0.14,0.83, 0.8]) #This is to make sure the plots are of the same size
        return fig, ax
    
    fig, ax = new_fig_ax(figsize=(10,6))
    for grp in ["SAA(n1=n2=n3)", "SAA(n1=n2^2=n3^2)"]:
        d = saa_data[grp]
        ax.scatter(d["x"], d["y_mse"], label=d["label"], marker=d["marker"], color=d["color"])
        if "line_mse" in d:
            ax.plot(x_line_range, d["line_mse"], color=d["color"], linestyle=d["style"])
            print(f"Slope of {grp} (MSE): {d['slope_mse']:.3f}")

    # Untruncated MLMC
    ax.scatter(x_untrunc, log_mse_untrunc, label="Untruncated MLMC", marker="^", color='royalblue')
    ax.plot(x_line_range, line_untr, linestyle="dashdot", color='royalblue')

    # Truncated MLMC
    ax.scatter(x_trunc, log_mse_trunc, label="Truncated MLMC", marker="p", color='darkorange')
    ax.plot(x_line_range, line_tr, color="#FF8C00", linestyle="solid")

    ax.legend(loc="upper right")
    ax.set_xlabel(r"$\log_{10}$(Total Sampling Cost)")
    ax.set_ylabel(r"$\log_{10}$(MSE)")
    ax.grid(True)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([-4, -3, -2, -1, 0])
    ax.set_xlim(1, 7)
    ax.set_ylim(-4, 0)

    mse_cost = os.path.join(save_dir, f"mse_vs_cost_{summary['timestamp']}.pdf")
    fig.savefig(mse_cost, format="pdf")
    plt.close(fig)

    # PLOT running avg
    fig, ax = new_fig_ax(figsize=(10,6))
    # Untruncated MLMC
    ax.plot(points, est_untr, label="Untruncated MLMC", color="blue", linestyle="dashdot")
    ax.fill_between(points, lower_untr, upper_untr, color="blue", alpha=0.3)

    # Truncated MLMC
    ax.plot(points, est_tr, label=f"Truncated MLMC", color="darkorange", linestyle="solid")
    ax.fill_between(points, lower_tr, upper_tr, color="darkorange", alpha=0.3)
    ax.axhline(true_val, color="black", linestyle="dotted", label="True value")

    ax.set_xlabel(r"Number of Trees ($n_1$)")
    ax.set_ylabel(r"Running Average ($\hat{F}(x)$)")
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_ylim(0.475, 0.675)
    ax.set_yticks([0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675])
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlim(5*10**(num_call_exponent-2), n_max)
    xticks = ax.get_xticks()  # Get the current tick locations
    xticks = sorted(set(xticks) | {5*10**(num_call_exponent-2)})  # Ensure 500 is included
    xticks = [tick for tick in xticks if tick != 0] # Remove 0 if it exists
    ax.set_xticks(xticks)
    running_avg = os.path.join(save_dir, f"running_avg_ci_{summary['timestamp']}.pdf")
    fig.savefig(running_avg, format="pdf")
    plt.close(fig)
    return slopes