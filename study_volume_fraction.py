#!/usr/bin/env python3
"""
study_volume_fraction.py
========================
Parametric study: influence of the volume fraction V₁ on the effective
elastic coefficients of a laminated composite, evaluated with PINNs and
compared against the analytical AHM solution.

Sweep: V₁ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
Material properties: E₁ = 206.74 GPa, ν₁ = 0.3  (fixed)
                     E₂ =  72.04 GPa, ν₂ = 0.35 (fixed)
Runs:  10 independent trainings per V₁ value (different seeds)

Output:
    parametric_vf/
    ├── vf_0.10/ … vf_0.90/   (per-run results.json inside)
    ├── data/
    │   ├── sweep_vf_all.csv
    │   ├── sweep_vf_statistics.csv
    │   └── sweep_vf_all.json
    └── figures/
        ├── ceff_vs_vf.pdf / .png
        └── relerr_vs_vf.pdf / .png

Usage:
    python study_volume_fraction.py
    python study_volume_fraction.py --n-runs 5 --quiet
    python study_volume_fraction.py --vf-values 0.2,0.5,0.8
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parametric_utils import (
    compute_stiffness_matrix,
    compute_all_exact,
    run_parameter_point,
    save_sweep_csv,
    save_sweep_statistics_csv,
    save_sweep_json,
    detect_sweep_progress,
    prompt_resume_or_restart,
    format_elapsed,
    BOLD, RESET, GREEN, BLUE, MAGENTA, CYAN, YELLOW,
    TRAIN_SEED,
)

# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_VF_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
E1, NU1 = 206.74, 0.3
E2, NU2 = 72.04, 0.35
DEFAULT_N_RUNS = 10
DEFAULT_OUTPUT_DIR = "parametric_vf"

# ══════════════════════════════════════════════════════════════════════════════
# Plot style (consistent with generate_paper_figures.py)
# ══════════════════════════════════════════════════════════════════════════════
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

# Colours per coefficient group
COLORS = {
    'R11_eff': '#2171B5',  'R22_eff': '#D94801',  'R33_eff': '#238B45',
    'R12_eff': '#6A51A3',  'R13_eff': '#E6550D',  'R23_eff': '#31A354',
    'R44_eff': '#2171B5',  'R55_eff': '#D94801',  'R66_eff': '#238B45',
}

MARKERS = {
    'R11_eff': 'o', 'R22_eff': 's', 'R33_eff': '^',
    'R12_eff': 'o', 'R13_eff': 's', 'R23_eff': '^',
    'R44_eff': 'o', 'R55_eff': 's', 'R66_eff': '^',
}

LABELS = {
    'R11_eff': r'$\~C_{11}$', 'R22_eff': r'$\~C_{22}$', 'R33_eff': r'$\~C_{33}$',
    'R12_eff': r'$\~C_{12}$', 'R13_eff': r'$\~C_{13}$', 'R23_eff': r'$\~C_{23}$',
    'R44_eff': r'$\~C_{44}$', 'R55_eff': r'$\~C_{55}$', 'R66_eff': r'$\~C_{66}$',
}


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _collect_series(all_results, prop_key):
    """Return arrays (vf_vals, means, stds, exacts) for one coefficient."""
    vf_vals, means, stds, exacts = [], [], [], []
    for res in all_results:
        if prop_key not in res['stats']:
            continue
        s = res['stats'][prop_key]
        vf_vals.append(res['param_value'])
        means.append(s['mean'])
        stds.append(s['std'])
        exacts.append(s['exact'])
    return (np.array(vf_vals), np.array(means),
            np.array(stds), np.array(exacts))


def fig_ceff_vs_vf(all_results, output_dir):
    """
    Three-panel figure: effective coefficients vs V₁.
    Panel a: diagonal (C11, C22, C33)
    Panel b: off-diagonal (C12, C13, C23)
    Panel c: shear (C44, C55, C66)
    """
    groups = [
        ('Diagonal coefficients',     ['R11_eff', 'R22_eff', 'R33_eff']),
        ('Off-diagonal coefficients',  ['R12_eff', 'R13_eff', 'R23_eff']),
        ('Shear coefficients',         ['R44_eff', 'R55_eff', 'R66_eff']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            vf, mean, std, exact = _collect_series(all_results, key)
            if len(vf) == 0:
                continue
            color = COLORS[key]
            marker = MARKERS[key]

            # AHM exact (dashed)
            ax.plot(vf, exact, '--', color=color, linewidth=1.3, alpha=0.7)

            # PINN mean ± std (markers + band)
            ax.fill_between(vf, mean - std, mean + std,
                            color=color, alpha=0.15)
            ax.plot(vf, mean, '-', color=color, marker=marker,
                    markersize=5, linewidth=1.5,
                    label=f"{LABELS[key]} PINN")
            # Invisible line just for the exact legend entry
            ax.plot([], [], '--', color=color, linewidth=1.3, alpha=0.7,
                    label=f"{LABELS[key]} AHM")

        ax.set_xlabel(r'Volume fraction $V_1$')
        ax.set_ylabel(r'Effective coefficient (GPa)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)
        ax.set_xlim(0.05, 0.95)

    fig.suptitle(
        r'Effective coefficients $\~C_{ij}$ vs. volume fraction $V_1$'
        f'\n(PINN mean ± std over {_n_runs_from(all_results)} runs vs. AHM exact)',
        fontsize=12, y=1.04,
    )
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(output_dir, f"ceff_vs_vf.{ext}")
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  {MAGENTA}Figure saved → {os.path.join(output_dir, 'ceff_vs_vf.pdf')}{RESET}")


def fig_relerr_vs_vf(all_results, output_dir):
    """
    Relative error (%) of the PINN mean vs. AHM exact for each coefficient,
    plotted against V₁.
    """
    # C66 excluded (trivial problem, zero error distorts log scale)
    prop_keys = ['R11_eff', 'R22_eff', 'R33_eff',
                 'R12_eff', 'R13_eff', 'R23_eff',
                 'R44_eff', 'R55_eff']

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in prop_keys:
        vf_vals, errs = [], []
        for res in all_results:
            if key not in res['stats']:
                continue
            s = res['stats'][key]
            vf_vals.append(res['param_value'])
            errs.append(s['rel_err_mean_pct'])
        if not vf_vals:
            continue

        ax.plot(vf_vals, errs, '-', marker=MARKERS[key], color=COLORS[key],
                markersize=5, linewidth=1.2, label=LABELS[key])

    ax.set_xlabel(r'Volume fraction $V_1$')
    ax.set_ylabel('Relative error of mean (%)')
    ax.set_title('PINN relative error vs. AHM exact')
    ax.legend(loc='best', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_xlim(0.05, 0.95)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(output_dir, f"relerr_vs_vf.{ext}")
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  {MAGENTA}Figure saved → {os.path.join(output_dir, 'relerr_vs_vf.pdf')}{RESET}")


def fig_boxplot_vs_vf(all_results, output_dir):
    """
    Box plot of PINN relative deviations (%) for each coefficient at each V₁.
    One panel per coefficient group.
    """
    groups = [
        ('Diagonal',     ['R11_eff', 'R22_eff', 'R33_eff']),
        ('Off-diagonal', ['R12_eff', 'R13_eff', 'R23_eff']),
        ('Shear',        ['R44_eff', 'R55_eff', 'R66_eff']),
    ]

    for group_name, keys in groups:
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5),
                                 sharey=True)
        if len(keys) == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            positions = []
            data_boxes = []
            for res in all_results:
                if key not in res['stats']:
                    continue
                s = res['stats'][key]
                exact = s['exact']
                vals = np.array(s['values'])
                rel_dev = (vals - exact) / exact * 100.0
                positions.append(res['param_value'])
                data_boxes.append(rel_dev)

            if not positions:
                continue

            bp = ax.boxplot(
                data_boxes, positions=positions,
                widths=0.035, patch_artist=True,
                boxprops=dict(facecolor='#B3CDE3', edgecolor='#2171B5', linewidth=1.0),
                medianprops=dict(color='#D94801', linewidth=1.5),
                whiskerprops=dict(color='#2171B5', linewidth=0.8),
                capprops=dict(color='#2171B5', linewidth=0.8),
                flierprops=dict(marker='o', markersize=3, markerfacecolor='#D94801',
                                markeredgecolor='#D94801', alpha=0.6),
            )
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
            ax.set_title(LABELS[key])
            ax.set_xlabel(r'$V_1$')
            if ax == axes[0]:
                ax.set_ylabel('Relative deviation from AHM (%)')

        fig.suptitle(f'{group_name} coefficients — distribution across runs',
                     fontsize=12, y=1.02)
        plt.tight_layout()
        tag = group_name.lower().replace('-', '')
        for ext in ('pdf', 'png'):
            path = os.path.join(output_dir, f"boxplot_vf_{tag}.{ext}")
            fig.savefig(path, dpi=300)
        plt.close(fig)

    print(f"  {MAGENTA}Box-plot figures saved → {output_dir}{RESET}")


def _n_runs_from(all_results):
    """Infer the number of runs from the first result entry."""
    if all_results and all_results[0]['all_runs']:
        return len(all_results[0]['all_runs'])
    return '?'


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Parametric study: volume fraction V₁"
    )
    parser.add_argument('--vf-values', type=str, default=None,
                        help='Comma-separated V₁ values (default: 0.1,0.2,…,0.9)')
    parser.add_argument('--n-runs', type=int, default=DEFAULT_N_RUNS,
                        help=f'Independent runs per V₁ value (default: {DEFAULT_N_RUNS})')
    parser.add_argument('--base-seed', type=int, default=TRAIN_SEED,
                        help=f'Base random seed (default: {TRAIN_SEED})')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Convergence tolerance %% (default: 0.05)')
    parser.add_argument('--max-cycles', type=int, default=100,
                        help='Max training cycles per problem (default: 100)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-iteration Adam/LBFGS logs')
    parser.add_argument('--figures-only', action='store_true',
                        help='Skip training; regenerate figures from existing data')
    args = parser.parse_args()

    vf_values = ([float(v) for v in args.vf_values.split(',')]
                 if args.vf_values else DEFAULT_VF_VALUES)

    base_dir = args.output_dir
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Compute material stiffness matrices (fixed for this study)
    R_mat1 = compute_stiffness_matrix(E1, NU1)
    R_mat2 = compute_stiffness_matrix(E2, NU2)

    if args.figures_only:
        # Reload results from JSON
        json_path = os.path.join(data_dir, 'sweep_vf_all.json')
        if not os.path.exists(json_path):
            print(f"{YELLOW}No data found at {json_path}. Run training first.{RESET}")
            return
        import json
        with open(json_path, 'r') as f:
            serialised = json.load(f)
        # Reconstruct the minimal structure expected by figure functions
        all_results = []
        for entry in serialised:
            vf1 = entry['VF1']
            exact = entry['exact']
            stats = entry['stats']
            all_results.append({
                'param_value': entry['V1'],
                'VF1': vf1,
                'exact': exact,
                'stats': stats,
                'all_runs': [],
            })
    else:
        # ── Check for previous progress ──
        param_labels = [f"vf_{vf:.2f}" for vf in vf_values]
        progress = detect_sweep_progress(
            base_dir, param_labels, args.n_runs, args.base_seed)

        if progress['has_progress']:
            prompt_resume_or_restart(progress, base_dir)

        # ── Run the parametric sweep ──
        print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"{BOLD}{CYAN}  PARAMETRIC STUDY — VOLUME FRACTION V₁{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"  V₁ values  : {vf_values}")
        print(f"  Runs/point : {args.n_runs}")
        print(f"  Materials  : E₁={E1}, ν₁={NU1} / E₂={E2}, ν₂={NU2}")
        print(f"  Output     : {base_dir}")
        print()

        global_t0 = time.perf_counter()
        all_results = []

        for i, vf in enumerate(vf_values, 1):
            label = f"vf_{vf:.2f}"
            print(f"\n  {BOLD}{BLUE}[{i}/{len(vf_values)}] V₁ = {vf:.2f}{RESET}")

            result = run_parameter_point(
                param_label=label,
                param_value=vf,
                VF1=vf,
                R_mat1=R_mat1,
                R_mat2=R_mat2,
                n_runs=args.n_runs,
                base_seed=args.base_seed,
                tolerance=args.tolerance,
                max_cycles=args.max_cycles,
                quiet=args.quiet,
                base_dir=base_dir,
            )
            all_results.append(result)

            # Incremental save after each V₁ value
            save_sweep_csv(all_results,
                           os.path.join(data_dir, 'sweep_vf_all.csv'),
                           param_col_name='V1')
            save_sweep_statistics_csv(all_results,
                                      os.path.join(data_dir, 'sweep_vf_statistics.csv'),
                                      param_col_name='V1')
            save_sweep_json(all_results,
                            os.path.join(data_dir, 'sweep_vf_all.json'),
                            param_col_name='V1')

        total_time = time.perf_counter() - global_t0
        print(f"\n{GREEN}Total sweep time: {format_elapsed(total_time)}{RESET}")

    # ── Generate figures ──
    print(f"\n{BOLD}Generating figures …{RESET}")
    fig_ceff_vs_vf(all_results, fig_dir)
    fig_relerr_vs_vf(all_results, fig_dir)
    fig_boxplot_vs_vf(all_results, fig_dir)

    print(f"\n{GREEN}✅ Volume-fraction study complete.  Results in: {base_dir}{RESET}")


if __name__ == '__main__':
    main()
