from collections import defaultdict
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times",
    "mathtext.it": "Times:italic",
    "mathtext.bf": "Times:bold",
    "axes.labelsize": 16,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb}\usepackage{times}",
})

SAA_PALETTE = [
    "#028A9B", "#FFD92F", "#E7298A", "#6B4E8C", "#DE5A5A", "#5D8C2A", "#BFA194", "#2E7DB0",
    "#B3B3B3", "#A6761D", "#1B9E77", "#7570B3", "#33A02C", "#F8CF9B",
]
MLMC_PALETTE = [
    "#F8A147", "#00A6FF", "#BF3F76", "#8C6D46", "#4A917C", "#C44536",
    "#1F78B4", "#E6AB02", "#A6D854", "#E5C494", "#B3B3B3", "#F8CF9B", "#33A02C",
]

LEGEND_FRAME_ALPHA = 0.80
LEGEND_FACE_COLOR = "white"
TRUE_VALUE_LEGEND_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
LEGEND_TITLE_FONT_SIZE = 14
LINE_WIDTH = 1.2


def mean_ci(a, z=1.96):
    """
    Compute mean and half-CI for array `a` of shape (T,) or (R,T).
    If only 1D (single run), returns the array itself and zeros (no CI).
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a, np.zeros_like(a)
    if a.shape[0] == 1:
        return a[0], np.zeros_like(a[0])
    mean = a.mean(axis=0)
    se = a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])
    return mean.squeeze(), (z * se).squeeze()


def group_by_plot_label(res_all, keys):
    """
    Groups runs by plot label and stacks the requested keys into arrays.
    """
    if not res_all:
        return []

    grouped = defaultdict(list)
    for r in res_all:
        plot_label = str(r.get("plot_label") or r.get("method") or "run")
        grouped[plot_label].append(r)

    out = []
    for plot_label, runs in grouped.items():
        item = {"plot_label": plot_label}
        for key in keys:
            item[key] = np.stack([r[key] for r in runs], axis=0)
        out.append(item)
    return out


def _simplify_setup_label(setup_label, method_name):
    text = str(setup_label).replace("$", "")
    compact = text.replace(" ", "")

    if method_name == "SAA":
        if "=" in compact:
            return compact.split("=")[-1]
        return compact.replace("SAA", "")

    mlmc_match = re.search(r"n_1=([^,]+),M=(\([^)]*\)),r=(\([^)]*\))", compact)
    if mlmc_match:
        n1 = mlmc_match.group(1)
        trunc = mlmc_match.group(2)
        rate = mlmc_match.group(3)
        return f"({n1},{trunc},{rate})"

    return compact.replace("MLMC", "")


def plot_all_three(res_saa_all, res_mlmc_all, title, outfile, true_vals=None):
    """
    Plot lambda, theta_1, and theta_2 against cumulative sample paths,
    horizontally with a shared top legend.
    """
    if not res_saa_all and not res_mlmc_all:
        return

    fig = plt.figure(figsize=(20, 5))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.09)
    theta_gs = outer_gs[0, 1].subgridspec(1, 2, wspace=0.08)
    ax_lam = fig.add_subplot(outer_gs[0, 0])
    ax_th1 = fig.add_subplot(theta_gs[0, 0])
    ax_th2 = fig.add_subplot(theta_gs[0, 1], sharey=ax_th1)

    grouped_saa = group_by_plot_label(
        res_saa_all,
        ["lambda_", "theta1", "theta2", "cost_history"],
    ) if res_saa_all else []
    grouped_mlmc = group_by_plot_label(
        res_mlmc_all,
        ["lambda_", "theta1", "theta2", "cost_history"],
    ) if res_mlmc_all else []

    saa_keys = [item["plot_label"] for item in grouped_saa]
    mlmc_keys = [item["plot_label"] for item in grouped_mlmc]

    saa_color_map = {
        key: SAA_PALETTE[i % len(SAA_PALETTE)]
        for i, key in enumerate(dict.fromkeys(saa_keys))
    }
    mlmc_color_map = {
        key: MLMC_PALETTE[i % len(MLMC_PALETTE)]
        for i, key in enumerate(dict.fromkeys(mlmc_keys))
    }

    x_min_target = 1e5

    def _plot_group(ax, y_key, grouped_items, color_map, linestyle):
        for item in grouped_items:
            setup_key = item["plot_label"]
            color = color_map[setup_key]
            y_matrix = np.asarray(item[y_key], dtype=float)
            x_matrix = np.asarray(item["cost_history"], dtype=float)
            min_len = min(y_matrix.shape[1], x_matrix.shape[1])
            y_matrix = y_matrix[:, :min_len]
            x_matrix = x_matrix[:, :min_len]

            mean_y, half_y = mean_ci(y_matrix)
            mean_x = np.mean(x_matrix, axis=0)
            positive_mask = mean_x > 0
            if not np.any(positive_mask):
                continue

            plot_x = mean_x[positive_mask]
            plot_y = mean_y[positive_mask]
            plot_half = half_y[positive_mask]
            if plot_x[0] > x_min_target:
                plot_x = np.insert(plot_x, 0, x_min_target)
                plot_y = np.insert(plot_y, 0, plot_y[0])
                plot_half = np.insert(plot_half, 0, plot_half[0])

            ax.plot(plot_x, plot_y, lw=LINE_WIDTH, ls=linestyle, color=color)
            ax.fill_between(plot_x, plot_y - plot_half, plot_y + plot_half, color=color, alpha=0.2)

    _plot_group(ax_lam, "lambda_", grouped_saa, saa_color_map, "-")
    _plot_group(ax_lam, "lambda_", grouped_mlmc, mlmc_color_map, "--")
    _plot_group(ax_th1, "theta1", grouped_saa, saa_color_map, "-")
    _plot_group(ax_th1, "theta1", grouped_mlmc, mlmc_color_map, "--")
    _plot_group(ax_th2, "theta2", grouped_saa, saa_color_map, "-")
    _plot_group(ax_th2, "theta2", grouped_mlmc, mlmc_color_map, "--")

    for ax in (ax_lam, ax_th1, ax_th2):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"\# Sample Paths")
        ax.set_xscale("log")

    positive_x = []
    for source in (res_saa_all or []) + (res_mlmc_all or []):
        if "cost_history" in source:
            vals = np.asarray(source["cost_history"], dtype=float)
            positive_x.extend(vals[vals > 0].tolist())
    if positive_x:
        x_min = x_min_target
        x_max = max(positive_x)
        for ax in (ax_lam, ax_th1, ax_th2):
            ax.set_xlim(x_min, x_max)

    ax_lam.set_ylabel(r"$\lambda$")
    ax_th1.set_ylabel(r"$\theta_1$")
    ax_th2.set_ylabel(r"$\theta_2$")
    ax_th2.tick_params(labelleft=False)
    ax_lam.set_ylim(5, 60)
    lam_ticks = [tick for tick in ax_lam.get_yticks() if 5 <= tick <= 60]
    ax_lam.set_yticks(sorted(set([5] + lam_ticks)))
    ax_th1.set_ylim(0, 0.8)
    ax_th2.set_ylim(0, 0.8)

    if true_vals is not None:
        true_specs = [
            (ax_lam, float(true_vals[0]), rf"True Optimum $\lambda$ ({float(true_vals[0]):.3f})"),
            (ax_th1, float(true_vals[1]), rf"True Optimum $\theta_1$ ({float(true_vals[1]):.3f})"),
            (ax_th2, float(true_vals[2]), rf"True Optimum $\theta_2$ ({float(true_vals[2]):.3f})"),
        ]
        for ax, y_val, label in true_specs:
            handle = ax.axhline(y=y_val, color="black", linestyle=":", linewidth=LINE_WIDTH, label=label)
            ax.legend(
                handles=[handle],
                frameon=True,
                facecolor=LEGEND_FACE_COLOR,
                framealpha=LEGEND_FRAME_ALPHA,
                fontsize=TRUE_VALUE_LEGEND_FONT_SIZE,
                loc="upper right",
            )

    saa_handles = [
        Line2D(
            [0],
            [0],
            color=saa_color_map[key],
            lw=LINE_WIDTH,
            ls="-",
            label=_simplify_setup_label(key, "SAA"),
        )
        for key in saa_color_map
    ]
    mlmc_handles = [
        Line2D(
            [0],
            [0],
            color=mlmc_color_map[key],
            lw=LINE_WIDTH,
            ls="--",
            label=_simplify_setup_label(key, "MLMC"),
        )
        for key in mlmc_color_map
    ]

    if saa_handles:
        legend_saa = fig.legend(
            handles=saa_handles,
            loc="upper center",
            bbox_to_anchor=(0.35, 0.99),
            ncol=max(1, len(saa_handles)),
            frameon=True,
            facecolor=LEGEND_FACE_COLOR,
            framealpha=LEGEND_FRAME_ALPHA,
            fontsize=LEGEND_FONT_SIZE,
            labelspacing=0.15,
            columnspacing=0.8,
            handlelength=2.0,
            handletextpad=0.4,
        )
        fig.add_artist(legend_saa)
        fig.text(
            0.35,
            0.99,
            r"SAA Setups $(n_1,n_2,n_3)$",
            ha="center",
            va="bottom",
            fontsize=LEGEND_TITLE_FONT_SIZE,
        )
        for line in legend_saa.get_lines():
            line.set_linewidth(1.5)

    if mlmc_handles:
        legend_mlmc = fig.legend(
            handles=mlmc_handles,
            loc="upper center",
            bbox_to_anchor=(0.70, 0.99),
            ncol=max(1, len(mlmc_handles)),
            frameon=True,
            facecolor=LEGEND_FACE_COLOR,
            framealpha=LEGEND_FRAME_ALPHA,
            fontsize=LEGEND_FONT_SIZE,
            labelspacing=0.15,
            columnspacing=0.8,
            handlelength=2.0,
            handletextpad=0.4,
        )
        fig.text(
            0.70,
            0.99,
            r"MLMC Setups $(n_1,M,r)$",
            ha="center",
            va="bottom",
            fontsize=LEGEND_TITLE_FONT_SIZE,
        )
        for line in legend_mlmc.get_lines():
            line.set_linewidth(1.5)

    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(outfile, format="pdf", facecolor="white", bbox_inches="tight")
    plt.close()
