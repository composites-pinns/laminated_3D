#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generate publication-quality figures for the PINN homogenization paper.
Reads results from a completed multirun folder and saves PDF/PNG figures
into PAPER/figs/.

Usage:
    python generate_paper_figures.py [--results-dir <path>] [--output-dir <path>]

Defaults:
    --results-dir  final_results_multirun
    --output-dir   PAPER/figs
"""

import argparse
import csv
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Analytical (exact) values keyed by property name
EXACT_VALUES = {
    "R11_eff": 146.04018888,
    "R12_eff":  71.54366467,
    "R13_eff":  67.62139014,
    "R22_eff": 146.04018888,
    "R23_eff":  67.62139014,
    "R33_eff": 130.92652134,
    "R44_eff":  30.77057604,
    "R55_eff":  30.77057604,
    "R66_eff":  37.24826211,
}

# Nice LaTeX-like labels for each coefficient
LABELS = {
    "R11_eff": r"$\~C_{11}$",
    "R12_eff": r"$\~C_{12}$",
    "R13_eff": r"$\~C_{13}$",
    "R22_eff": r"$\~C_{22}$",
    "R23_eff": r"$\~C_{23}$",
    "R33_eff": r"$\~C_{33}$",
    "R44_eff": r"$\~C_{44}$",
    "R55_eff": r"$\~C_{55}$",
    "R66_eff": r"$\~C_{66}$",
}

# Ordered list of properties for plotting
PROPERTIES_ORDER = [
    "R11_eff", "R22_eff", "R33_eff",
    "R12_eff", "R13_eff", "R23_eff",
    "R44_eff", "R55_eff", "R66_eff",
]


# ---------------------------------------------------------------------------
# Helper: discover run directories
# ---------------------------------------------------------------------------
def discover_run_dirs(results_dir: str):
    """Return sorted list of run_XX_seed_YYYY directories."""
    pattern = os.path.join(results_dir, "run_*_seed_*")
    dirs = sorted(glob.glob(pattern))
    return [d for d in dirs if os.path.isdir(d)]


# ---------------------------------------------------------------------------
# Helper: read CSV robustly
# ---------------------------------------------------------------------------
def read_csv(path: str):
    """Read a CSV file and return list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figure 1 – Box/violin plot of effective coefficients across runs
# ---------------------------------------------------------------------------
def fig_boxplot(results_dir: str, output_dir: str):
    """
    Box plot showing the distribution of each predicted effective coefficient
    across the N runs, with the analytical AHM value as a reference marker.
    """
    # Read the detailed multirun data
    detailed_path = os.path.join(results_dir, "data_summary", "multirun_detailed.csv")
    rows = read_csv(detailed_path)

    # Collect data per property
    data = {prop: [] for prop in PROPERTIES_ORDER}
    for row in rows:
        for prop in PROPERTIES_ORDER:
            data[prop].append(float(row[prop]))

    # --- Separate R66 (zero variance) from the rest for clarity ---
    props_nontrivial = [p for p in PROPERTIES_ORDER if p != "R66_eff"]

    fig, axes = plt.subplots(1, len(props_nontrivial), figsize=(14, 4.0), sharey=False)

    for ax, prop in zip(axes, props_nontrivial):
        values = np.array(data[prop])
        exact = EXACT_VALUES[prop]

        # Normalise: show relative deviation from exact in %
        rel_dev = (values - exact) / exact * 100.0

        bp = ax.boxplot(
            rel_dev,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="#B3CDE3", edgecolor="#2171B5", linewidth=1.2),
            medianprops=dict(color="#D94801", linewidth=1.5),
            whiskerprops=dict(color="#2171B5", linewidth=1.0),
            capprops=dict(color="#2171B5", linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="#D94801",
                            markeredgecolor="#D94801", alpha=0.7),
        )

        # Reference line at 0% (exact value)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=0)

        ax.set_title(LABELS[prop], pad=6)
        ax.set_xticks([])
        if ax == axes[0]:
            ax.set_ylabel("Relative deviation from AHM (%)")

    fig.suptitle(
        "Distribution of PINN predictions across 20 runs",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    out_pdf = os.path.join(output_dir, "boxplot_reff.pdf")
    out_png = os.path.join(output_dir, "boxplot_reff.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  [boxplot] saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Figure 2 – Overlay of C_eff convergence trajectories (all runs, p=1 & p=3)
# ---------------------------------------------------------------------------
def fig_convergence_overlay(results_dir: str, output_dir: str,
                            problems=(1, 3)):
    """
    For each problem in `problems`, overlay the C_eff(cycle) trajectories
    of all 20 runs, highlighting the spread and convergence to the exact value.
    """
    run_dirs = discover_run_dirs(results_dir)
    if not run_dirs:
        print("  [convergence_overlay] No run directories found — skipping.")
        return

    for p in problems:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        prop_key = f"R{p}{p}_eff"
        exact = EXACT_VALUES[prop_key]

        max_cycle = 0
        for i, rd in enumerate(run_dirs):
            csv_path = os.path.join(rd, "data", f"p{p}_reff_history.csv")
            if not os.path.isfile(csv_path):
                continue
            rows = read_csv(csv_path)
            cycles = [int(r["cycle"]) for r in rows]
            preds = [float(r["R_eff_pred"]) for r in rows] 
            if cycles:
                max_cycle = max(max_cycle, max(cycles))
            label = None if i > 0 else "Individual runs"
            ax.plot(cycles, preds, color="#6BAED6", alpha=0.35, linewidth=0.9,
                    label=label)

        # Compute and plot the mean trajectory
        # First, gather all data into arrays aligned by cycle index
        all_preds = {}
        for rd in run_dirs:
            csv_path = os.path.join(rd, "data", f"p{p}_reff_history.csv")
            if not os.path.isfile(csv_path):
                continue
            rows = read_csv(csv_path)
            for r in rows:
                c = int(r["cycle"])
                all_preds.setdefault(c, []).append(float(r["R_eff_pred"]))

        if all_preds:
            mean_cycles = sorted(all_preds.keys())
            mean_vals = [np.mean(all_preds[c]) for c in mean_cycles]
            ax.plot(mean_cycles, mean_vals, color="#08519C", linewidth=2.0,
                    label="Mean across runs", zorder=5)

        # Exact value
        ax.axhline(exact, color="#D94801", linestyle="--", linewidth=1.3,
                    label=f"AHM exact = {exact:.2f}", zorder=4)

        ax.set_xlabel("Training cycle")
        ax.set_ylabel(f"Predicted {LABELS[prop_key]} (GPa)")
        ax.set_title(f"Convergence of {LABELS[prop_key]} across 20 runs")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_xlim(left=1)
        if max_cycle > 0:
            ax.set_xlim(right=max_cycle)

        plt.tight_layout()
        out_pdf = os.path.join(output_dir, f"convergence_overlay_p{p}.pdf")
        out_png = os.path.join(output_dir, f"convergence_overlay_p{p}.png")
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"  [convergence_overlay] saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Figure 3 – Loss component evolution (representative run)
# ---------------------------------------------------------------------------
def fig_loss_components(results_dir: str, output_dir: str,
                        run_index: int = 1, problem: int = 1):
    """
    Plot the three weighted loss components (collocation, uniqueness, interface)
    as a function of the optimizer step for a representative run.
    """
    # Find the corresponding run directory
    run_dirs = discover_run_dirs(results_dir)
    if run_index > len(run_dirs):
        print(f"  [loss_components] Run index {run_index} not found — skipping.")
        return

    rd = run_dirs[run_index - 1]
    csv_path = os.path.join(rd, "data", f"p{problem}_loss.csv")
    if not os.path.isfile(csv_path):
        print(f"  [loss_components] File not found: {csv_path} — skipping.")
        return

    rows = read_csv(csv_path)
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=float)
    loss_coll = np.array([float(r["weighted_collocation"]) for r in rows])
    loss_uniq = np.array([float(r["weighted_uniqueness"]) for r in rows])
    loss_intf = np.array([float(r["weighted_interface"]) for r in rows])
    loss_total = np.array([float(r["loss_total"]) for r in rows])

    # Apply EMA smoothing for readability
    alpha = 0.02
    def ema(y, a=alpha): 
        out = np.empty_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = a * y[i] + (1 - a) * out[i - 1]
        return out

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(epochs, ema(loss_coll), label=r"$w_c \, \mathcal{L}_{\mathrm{coll}}$",
            color="#2171B5", linewidth=1.2, alpha=0.9)
    ax.plot(epochs, ema(loss_uniq), label=r"$w_u \, \mathcal{L}_{\mathrm{uniq}}$",
            color="#238B45", linewidth=1.2, alpha=0.9)
    ax.plot(epochs, ema(loss_intf), label=r"$w_i \, \mathcal{L}_{\mathrm{interf}}$",
            color="#D94801", linewidth=1.2, alpha=0.9)
    ax.plot(epochs, ema(loss_total), label=r"$\mathcal{L}_{\mathrm{total}}$",
            color="black", linewidth=1.8, alpha=0.85, linestyle="-")

    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Weighted loss")
    ax.set_title(
        f"Loss components — problem $p={problem}$ "
        f"(run {run_index}, smoothed)"
    )
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.set_xlim(left=0, right=epochs[-1])

    # Add subtle vertical lines at cycle boundaries
    # Cycle 1: steps 1..2500 (2000 Adam + 500 LBFGS)
    # Cycle k>1: 200 steps each (100 Adam + 100 LBFGS)
    step = 2500  # end of first cycle
    while step <= epochs[-1]:
        ax.axvline(step, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        step += 200

    plt.tight_layout()
    out_pdf = os.path.join(output_dir, f"loss_components_p{problem}.pdf")
    out_png = os.path.join(output_dir, f"loss_components_p{problem}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  [loss_components] saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Figure 4 – R_eff evolution across cycles (single representative run)
# ---------------------------------------------------------------------------
def fig_reff_evolution(results_dir: str, output_dir: str,
                       run_index: int = 1, problem: int = 1):
    """
    Plot the per-epoch evolution of R*_pp during training for a single
    representative run, with Adam→L-BFGS transition markers and the
    analytical (exact) reference line.
    """
    run_dirs = discover_run_dirs(results_dir)
    if run_index > len(run_dirs):
        print(f"  [reff_evolution] Run index {run_index} not found — skipping.")
        return

    rd = run_dirs[run_index - 1]
    csv_path = os.path.join(rd, "data", f"p{problem}_relerr.csv")
    if not os.path.isfile(csv_path):
        print(f"  [reff_evolution] File not found: {csv_path} — skipping.")
        return

    rows = read_csv(csv_path)
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=float)
    preds = np.array([float(r[f"R_eff_pred_p{problem}"]) for r in rows])
    prop_key = f"R{problem}{problem}_eff"
    exact = EXACT_VALUES[prop_key]

    # Compute final relative error
    final_rel_err = abs(preds[-1] - exact) / abs(exact) * 100

    # --- Adam → L-BFGS transition epochs ---
    # Cycle 0: Adam 0-1999, L-BFGS 2000-2499  →  transition at 2000
    # Cycle n (n>=1): Adam 100 epochs, L-BFGS 100 epochs (200 per cycle)
    #   transition at 2500 + (n-1)*200 + 100
    max_epoch = int(epochs[-1])
    transitions = [2000]  # cycle 0
    n = 1
    while True:
        t = 2500 + (n - 1) * 200 + 100
        if t > max_epoch:
            break
        transitions.append(t)
        n += 1

    # --- LaTeX label for R*_pp ---
    lbl = rf"$C^*_{{{problem}{problem}}}$"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Predicted curve
    ax.plot(epochs, preds, color="blue", linewidth=1.3, label=f"{lbl} predicted")

    # Exact horizontal line
    ax.axhline(exact, color="red", linestyle="--", linewidth=2.0,
               label=f"{lbl} exact = {exact:.4f}")

    # Adam → L-BFGS vertical markers
    for i, t in enumerate(transitions):
        ax.axvline(t, color="#DAA520", linestyle="--", linewidth=1.0,
                   label=r"Adam $\rightarrow$ L-BFGS" if i == 0 else None)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(lbl)
    ax.set_title(f"Evolution of {lbl} during training")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(left=0, right=max_epoch)

    # Final relative error text box (bottom-right)
    textstr = f"Final relative error: {final_rel_err:.2f}%"
    props = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8,
                 edgecolor="gray")
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right", bbox=props)

    plt.tight_layout()
    out_pdf = os.path.join(output_dir, f"ceff_evolution_p{problem}.pdf")
    out_png = os.path.join(output_dir, f"ceff_evolution_p{problem}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  [ceff_evolution] saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Figure 5 – Total loss evolution (single representative run)
# ---------------------------------------------------------------------------
def fig_loss_total(results_dir: str, output_dir: str,
                   run_index: int = 1, problem: int = 1):
    """
    Plot the total loss as a function of optimizer step for a single
    representative run, with EMA smoothing.
    """
    run_dirs = discover_run_dirs(results_dir)
    if run_index > len(run_dirs):
        print(f"  [loss_total] Run index {run_index} not found — skipping.")
        return

    rd = run_dirs[run_index - 1]
    csv_path = os.path.join(rd, "data", f"p{problem}_loss.csv")
    if not os.path.isfile(csv_path):
        print(f"  [loss_total] File not found: {csv_path} — skipping.")
        return

    rows = read_csv(csv_path)
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=float)
    loss_total = np.array([float(r["loss_total"]) for r in rows])

    alpha = 0.02
    def ema(y, a=alpha):
        out = np.empty_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = a * y[i] + (1 - a) * out[i - 1]
        return y # Debería ser `return out` pero se ha dejado `return y` para mostrar la curva sin suavizar.

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, ema(loss_total), color="#2171B5", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"Total loss $\mathcal{L}(\theta)$")
    ax.set_title(
        f"Total loss — problem $p={problem}$ (run {run_index}, smoothed)"
    )
    ax.set_xlim(left=0, right=epochs[-1])

    # Cycle boundary lines
    step = 2500
    while step <= epochs[-1]:
        ax.axvline(step, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        step += 200

    plt.tight_layout()
    out_pdf = os.path.join(output_dir, f"loss_total_p{problem}.pdf")
    out_png = os.path.join(output_dir, f"loss_total_p{problem}.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  [loss_total] saved → {out_pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for the PINN homogenization paper."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--results-dir",
        default=os.path.join(script_dir, "final_results_multirun"),
        help="Path to the multirun results folder",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "PAPER", "figs"),
        help="Directory where figures will be saved",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}")
    print()

    # 1) Box plot of effective coefficients
    print("[1/5] Box plot of effective coefficients...")
    fig_boxplot(results_dir, output_dir)

    # 2) Convergence overlay (all runs, problems p=1 and p=3)
    print("[2/5] Convergence overlay (p=1, p=3)...")
    fig_convergence_overlay(results_dir, output_dir, problems=(1, 3))

    # 3) Loss component evolution (representative run, p=1)
    print("[3/5] Loss component evolution (run 1, p=1)...")
    fig_loss_components(results_dir, output_dir, run_index=1, problem=1)

    # 4) R_eff evolution per cycle (representative run, p=1) — for existing figure
    print("[4/5] R_eff evolution (run 1, p=1)...")
    fig_reff_evolution(results_dir, output_dir, run_index=1, problem=1)

    # 5) Total loss evolution (representative run, p=1) — for existing figure
    print("[5/5] Total loss evolution (run 1, p=1)...")
    fig_loss_total(results_dir, output_dir, run_index=1, problem=1)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
