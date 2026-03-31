#!/usr/bin/env python3
"""
study_material_contrast.py
==========================
Parametric study: influence of the elastic-modulus contrast E₁/E₂ on the
effective elastic coefficients of a laminated composite, evaluated with
PINNs and compared against the analytical AHM solution.

Sweep: E₁/E₂ ∈ {1.5, 2, 5, 10, 25, 50, 100}
       E₂ = 72.04 GPa fixed, E₁ = ratio × E₂
       ν₁ = 0.3, ν₂ = 0.35 (fixed)
Volume fraction: V₁ = 0.2  (fixed)
Runs:  10 independent trainings per ratio (different seeds)

Output:
    parametric_contrast_material/
    ├── ratio_001.50/ … ratio_100.00/  (per-run results.json inside)
    ├── data/
    │   ├── sweep_contrast_all.csv
    │   ├── sweep_contrast_statistics.csv
    │   └── sweep_contrast_all.json
    └── figures/
        ├── ceff_vs_contrast.pdf / .png
        └── relerr_vs_contrast.pdf / .png

Usage:
    python study_material_contrast.py
    python study_material_contrast.py --n-runs 5 --quiet
    python study_material_contrast.py --ratios 2,10,50
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
DEFAULT_RATIOS = [1.5, 2, 5, 10, 25, 50, 100]
E2_FIXED = 72.04
NU1 = 0.3
NU2 = 0.35
VF1_FIXED = 0.2
DEFAULT_N_RUNS = 10
DEFAULT_OUTPUT_DIR = "parametric_contrast_material"

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

# Colours per coefficient (same palette as VF study for consistency)
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
    """Return arrays (ratio_vals, means, stds, exacts) for one coefficient."""
    rv, means, stds, exacts = [], [], [], []
    for res in all_results:
        if prop_key not in res['stats']:
            continue
        s = res['stats'][prop_key]
        rv.append(res['param_value'])
        means.append(s['mean'])
        stds.append(s['std'])
        exacts.append(s['exact'])
    return np.array(rv), np.array(means), np.array(stds), np.array(exacts)


def fig_ceff_vs_contrast(all_results, output_dir):
    """
    Three-panel figure: effective coefficients vs E₁/E₂ (log-scale x-axis).
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
            rv, mean, std, exact = _collect_series(all_results, key)
            if len(rv) == 0:
                continue
            color = COLORS[key]
            marker = MARKERS[key]

            # AHM exact (dashed)
            ax.plot(rv, exact, '--', color=color, linewidth=1.3, alpha=0.7)

            # PINN mean ± std
            ax.fill_between(rv, mean - std, mean + std,
                            color=color, alpha=0.15)
            ax.plot(rv, mean, '-', color=color, marker=marker,
                    markersize=5, linewidth=1.5,
                    label=f"{LABELS[key]} PINN")
            ax.plot([], [], '--', color=color, linewidth=1.3, alpha=0.7,
                    label=f"{LABELS[key]} AHM")

        ax.set_xscale('log')
        ax.set_xlabel(r'$E_1 / E_2$')
        ax.set_ylabel('Effective coefficient (GPa)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)

    n_runs = _n_runs_from(all_results)
    fig.suptitle(
        r'Effective coefficients $\~C_{ij}$ vs. material contrast $E_1/E_2$'
        f'\n(PINN mean ± std over {n_runs} runs vs. AHM exact,'
        f' $V_1 = {VF1_FIXED}$)',
        fontsize=12, y=1.04,
    )
    plt.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(output_dir, f"ceff_vs_contrast.{ext}")
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  {MAGENTA}Figure saved → "
          f"{os.path.join(output_dir, 'ceff_vs_contrast.pdf')}{RESET}")


def fig_relerr_vs_contrast(all_results, output_dir):
    """
    Relative error (%) of the PINN mean vs. AHM exact for each coefficient,
    plotted against E₁/E₂ (log scale).  C66 is excluded (trivial, zero error).
    """
    prop_keys = ['R11_eff', 'R22_eff', 'R33_eff',
                 'R12_eff', 'R13_eff', 'R23_eff',
                 'R44_eff', 'R55_eff']

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in prop_keys:
        rv_vals, errs = [], []
        for res in all_results:
            if key not in res['stats']:
                continue
            s = res['stats'][key]
            rv_vals.append(res['param_value'])
            errs.append(s['rel_err_mean_pct'])
        if not rv_vals:
            continue

        ax.plot(rv_vals, errs, '-', marker=MARKERS[key], color=COLORS[key],
                markersize=5, linewidth=1.2, label=LABELS[key])

    ax.set_xscale('log')
    ax.set_xlabel(r'$E_1 / E_2$')
    ax.set_ylabel('Relative error of mean (%)')
    ax.set_title(f'PINN relative error vs. AHM exact ($V_1 = {VF1_FIXED}$)')
    ax.legend(loc='best', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(output_dir, f"relerr_vs_contrast.{ext}")
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  {MAGENTA}Figure saved → "
          f"{os.path.join(output_dir, 'relerr_vs_contrast.pdf')}{RESET}")


def fig_boxplot_vs_contrast(all_results, output_dir):
    """
    Box plot of PINN relative deviations (%) for each coefficient at each E₁/E₂.
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
            # Collect per-ratio box data
            x_positions = []
            data_boxes = []
            x_labels = []
            for res in all_results:
                if key not in res['stats']:
                    continue
                s = res['stats'][key]
                exact = s['exact']
                vals = np.array(s['values'])
                rel_dev = (vals - exact) / exact * 100.0
                x_positions.append(len(data_boxes))
                data_boxes.append(rel_dev)
                x_labels.append(f"{res['param_value']:.4g}")

            if not data_boxes:
                continue

            bp = ax.boxplot(
                data_boxes, positions=list(range(len(data_boxes))),
                widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='#B3CDE3', edgecolor='#2171B5',
                              linewidth=1.0),
                medianprops=dict(color='#D94801', linewidth=1.5),
                whiskerprops=dict(color='#2171B5', linewidth=0.8),
                capprops=dict(color='#2171B5', linewidth=0.8),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor='#D94801',
                                markeredgecolor='#D94801', alpha=0.6),
            )
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8,
                       zorder=0)
            ax.set_xticks(list(range(len(data_boxes))))
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlabel(r'$E_1/E_2$')
            ax.set_title(LABELS[key])
            if ax == axes[0]:
                ax.set_ylabel('Relative deviation from AHM (%)')

        fig.suptitle(
            f'{group_name} coefficients — distribution across runs',
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        tag = group_name.lower().replace('-', '')
        for ext in ('pdf', 'png'):
            path = os.path.join(output_dir, f"boxplot_contrast_{tag}.{ext}")
            fig.savefig(path, dpi=300)
        plt.close(fig)

    print(f"  {MAGENTA}Box-plot figures saved → {output_dir}{RESET}")


def _n_runs_from(all_results):
    if all_results and all_results[0].get('all_runs'):
        return len(all_results[0]['all_runs'])
    return '?'


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Parametric study: material contrast E₁/E₂"
    )
    parser.add_argument('--ratios', type=str, default=None,
                        help='Comma-separated E₁/E₂ ratios '
                             '(default: 1.5,2,5,10,25,50,100)')
    parser.add_argument('--n-runs', type=int, default=DEFAULT_N_RUNS,
                        help=f'Independent runs per ratio (default: {DEFAULT_N_RUNS})')
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

    ratios = ([float(v) for v in args.ratios.split(',')]
              if args.ratios else DEFAULT_RATIOS)

    base_dir = args.output_dir
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Stiffness matrix for material 2 (fixed)
    R_mat2 = compute_stiffness_matrix(E2_FIXED, NU2)

    if args.figures_only:
        json_path = os.path.join(data_dir, 'sweep_contrast_all.json')
        if not os.path.exists(json_path):
            print(f"{YELLOW}No data found at {json_path}. Run training first.{RESET}")
            return
        import json
        with open(json_path, 'r') as f:
            serialised = json.load(f)
        all_results = []
        for entry in serialised:
            all_results.append({
                'param_value': entry['E1_over_E2'],
                'VF1': entry['VF1'],
                'exact': entry['exact'],
                'stats': entry['stats'],
                'all_runs': [],
            })
    else:
        # ── Check for previous progress ──
        param_labels = [f"ratio_{r:06.2f}" for r in ratios]
        progress = detect_sweep_progress(
            base_dir, param_labels, args.n_runs, args.base_seed)

        if progress['has_progress']:
            prompt_resume_or_restart(progress, base_dir)

        # ── Run the parametric sweep ──
        print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"{BOLD}{CYAN}  PARAMETRIC STUDY — MATERIAL CONTRAST E₁/E₂{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
        print(f"  Ratios     : {ratios}")
        print(f"  E₂ (fixed) : {E2_FIXED} GPa")
        print(f"  ν₁ / ν₂    : {NU1} / {NU2}")
        print(f"  V₁ (fixed) : {VF1_FIXED}")
        print(f"  Runs/point : {args.n_runs}")
        print(f"  Output     : {base_dir}")
        print()

        global_t0 = time.perf_counter()
        all_results = []

        for i, ratio in enumerate(ratios, 1):
            E1_current = ratio * E2_FIXED
            R_mat1 = compute_stiffness_matrix(E1_current, NU1)

            label = f"ratio_{ratio:06.2f}"
            print(f"\n  {BOLD}{BLUE}[{i}/{len(ratios)}] E₁/E₂ = {ratio}"
                  f"  (E₁ = {E1_current:.2f} GPa){RESET}")

            result = run_parameter_point(
                param_label=label,
                param_value=ratio,
                VF1=VF1_FIXED,
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

            # Incremental save
            save_sweep_csv(all_results,
                           os.path.join(data_dir, 'sweep_contrast_all.csv'),
                           param_col_name='E1_over_E2')
            save_sweep_statistics_csv(
                all_results,
                os.path.join(data_dir, 'sweep_contrast_statistics.csv'),
                param_col_name='E1_over_E2')
            save_sweep_json(all_results,
                            os.path.join(data_dir, 'sweep_contrast_all.json'),
                            param_col_name='E1_over_E2')

        total_time = time.perf_counter() - global_t0
        print(f"\n{GREEN}Total sweep time: {format_elapsed(total_time)}{RESET}")

    # ── Generate figures ──
    print(f"\n{BOLD}Generating figures …{RESET}")
    fig_ceff_vs_contrast(all_results, fig_dir)
    fig_relerr_vs_contrast(all_results, fig_dir)
    fig_boxplot_vs_contrast(all_results, fig_dir)

    print(f"\n{GREEN}✅ Material-contrast study complete.  Results in: {base_dir}{RESET}")


if __name__ == '__main__':
    main()
