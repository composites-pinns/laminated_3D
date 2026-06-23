#!/usr/bin/env python3
"""
plot_from_data.py
=================
Standalone script to generate focused diagnostic plots from the saved
parametric-study JSON data (no training needed).

Reads from:
  - parametric_contrast_material/data/sweep_contrast_all.json
  - parametric_contrast_vf/data/sweep_vf_all.json

Usage examples:
  # Deep-dive into C_33 vs material contrast
  python plot_from_data.py --study contrast --coeff R33_eff

  # Compare all diagonal coefficients from VF study
  python plot_from_data.py --study vf --coeff R11_eff,R22_eff,R33_eff

  # All 9 coefficients, contrast study, with box plots
  python plot_from_data.py --study contrast --coeff all --boxplot

  # Single coefficient with individual run scatter points
  python plot_from_data.py --study contrast --coeff R33_eff --scatter

  # Custom output directory
  python plot_from_data.py --study contrast --coeff R33_eff -o my_figs/

  # Convergence history: prediction vs exact across training cycles
  python plot_from_data.py --study contrast --coeff R33_eff --history
"""

import argparse
import json
import os
import sys

import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    'R11_eff': '#2171B5', 'R22_eff': '#D94801', 'R33_eff': '#238B45',
    'R12_eff': '#6A51A3', 'R13_eff': '#E6550D', 'R23_eff': '#31A354',
    'R44_eff': '#2171B5', 'R55_eff': '#D94801', 'R66_eff': '#238B45',
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

ALL_KEYS = ['R11_eff', 'R22_eff', 'R33_eff', 'R12_eff', 'R13_eff', 'R23_eff',
            'R44_eff', 'R55_eff', 'R66_eff']

# Mapping from coefficient key to the problem index p (1..6) in results.json
# Cross terms (R12, R13, R23) are extracted post-training and have no cycle history.
KEY_TO_PROBLEM = {
    'R11_eff': 1, 'R22_eff': 2, 'R33_eff': 3,
    'R44_eff': 4, 'R55_eff': 5, 'R66_eff': 6,
}

STUDY_CONFIG = {
    'contrast': {
        'json': 'parametric_contrast_material/data/sweep_contrast_all.json',
        'base_dir': 'parametric_contrast_material',
        'param_col': 'E1_over_E2',
        'param_fmt': 'ratio_{:06.2f}',
        'xlabel': r'$E_1 / E_2$',
        'logx': True,
        'title_tag': 'material contrast',
    },
    'vf': {
        'json': 'parametric_contrast_vf/data/sweep_vf_all.json',
        'base_dir': 'parametric_contrast_vf',
        'param_col': 'V1',
        'param_fmt': 'vf_{:.2f}',
        'xlabel': r'$V_1$',
        'logx': False,
        'title_tag': 'volume fraction',
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(study: str):
    cfg = STUDY_CONFIG[study]
    path = cfg['json']
    if not os.path.exists(path):
        sys.exit(f"ERROR: data file not found: {path}")
    with open(path, 'r') as f:
        raw = json.load(f)
    return raw, cfg


def extract_series(data, cfg, key):
    """Return (x_vals, means, stds, exacts, all_values_per_point)."""
    xs, means, stds, exacts, all_vals = [], [], [], [], []
    for entry in data:
        if key not in entry['stats']:
            continue
        s = entry['stats'][key]
        xs.append(entry[cfg['param_col']])
        means.append(s['mean'])
        stds.append(s['std'])
        exacts.append(s['exact'])
        all_vals.append(s['values'])
    return (np.array(xs), np.array(means), np.array(stds),
            np.array(exacts), all_vals)


# ── Figure: PINN vs AHM absolute values ─────────────────────────────────────

def fig_absolute(data, cfg, keys, output_dir, scatter=False):
    n = len(keys)
    if n == 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        axes = [ax]
    else:
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        axes = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        x, mean, std, exact, all_vals = extract_series(data, cfg, key)
        if len(x) == 0:
            continue
        color = COLORS[key]
        marker = MARKERS[key]

        # AHM exact
        ax.plot(x, exact, '--', color='gray', linewidth=1.5, alpha=0.8,
                label='AHM exact', zorder=3)
        # PINN mean ± std
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        ax.plot(x, mean, '-', color=color, marker=marker, markersize=6,
                linewidth=1.8, label='PINN mean ± std', zorder=4)

        if scatter:
            for j, vals in enumerate(all_vals):
                ax.scatter([x[j]] * len(vals), vals, s=12, color=color,
                           alpha=0.35, zorder=2, edgecolors='none')

        if cfg['logx']:
            ax.set_xscale('log')
        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel('Effective coefficient (GPa)')
        ax.set_title(LABELS[key])
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)

    # Hide extra axes
    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)

    tag = '_'.join(keys)
    fig.suptitle(
        f'PINN vs AHM — {cfg["title_tag"]}',
        fontsize=13, y=1.02)
    plt.tight_layout()
    fname = f"abs_{tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved → {os.path.join(output_dir, fname + '.pdf')}")


# ── Figure: relative error of mean ───────────────────────────────────────────

def fig_relerr(data, cfg, keys, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    for key in keys:
        x, mean, std, exact, _ = extract_series(data, cfg, key)
        if len(x) == 0:
            continue
        rel_err = np.abs(mean - exact) / np.abs(exact) * 100.0
        ax.plot(x, rel_err, '-', marker=MARKERS[key], color=COLORS[key],
                markersize=5, linewidth=1.2, label=LABELS[key])

    if cfg['logx']:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(cfg['xlabel'])
    ax.set_ylabel('Relative error of mean (%)')
    ax.set_title(f'PINN relative error vs AHM — {cfg["title_tag"]}')
    ax.legend(loc='best', fontsize=8, ncol=min(3, len(keys)), framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.6)
    plt.tight_layout()

    tag = '_'.join(keys)
    fname = f"relerr_{tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved → {os.path.join(output_dir, fname + '.pdf')}")


# ── Figure: box plot of relative deviations ──────────────────────────────────

def fig_boxplot(data, cfg, keys, output_dir):
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True,
                             squeeze=False)
    axes = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        x, mean, std, exact, all_vals = extract_series(data, cfg, key)
        if len(x) == 0:
            continue

        data_boxes = []
        x_labels = []
        for j in range(len(x)):
            vals = np.array(all_vals[j])
            rel_dev = (vals - exact[j]) / exact[j] * 100.0
            data_boxes.append(rel_dev)
            x_labels.append(f"{x[j]:.4g}")

        bp = ax.boxplot(
            data_boxes, positions=list(range(len(data_boxes))),
            widths=0.5, patch_artist=True,
            boxprops=dict(facecolor='#B3CDE3', edgecolor='#2171B5', linewidth=1.0),
            medianprops=dict(color='#D94801', linewidth=1.5),
            whiskerprops=dict(color='#2171B5', linewidth=0.8),
            capprops=dict(color='#2171B5', linewidth=0.8),
            flierprops=dict(marker='o', markersize=3,
                            markerfacecolor='#D94801',
                            markeredgecolor='#D94801', alpha=0.6),
        )
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
        ax.set_xticks(list(range(len(data_boxes))))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel(cfg['xlabel'])
        ax.set_title(LABELS[key])
        if i == 0:
            ax.set_ylabel('Relative deviation from AHM (%)')

    fig.suptitle(
        f'Distribution across runs — {cfg["title_tag"]}',
        fontsize=12, y=1.02)
    plt.tight_layout()

    tag = '_'.join(keys)
    fname = f"boxplot_{tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved → {os.path.join(output_dir, fname + '.pdf')}")


# ── Figure: convergence history (per-run scatter + mean) ─────────────────────

def fig_individual_runs(data, cfg, keys, output_dir):
    """For each coefficient, plot each individual run value as a dot."""
    n = len(keys)
    if n == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes = [ax]
    else:
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        axes = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        x, mean, std, exact, all_vals = extract_series(data, cfg, key)
        if len(x) == 0:
            continue
        color = COLORS[key]

        # AHM exact line
        ax.plot(x, exact, 'k--', linewidth=1.5, label='AHM exact', zorder=5)

        # Individual run scatter
        for j in range(len(x)):
            vals = all_vals[j]
            ax.scatter([x[j]] * len(vals), vals, s=20, color=color,
                       alpha=0.5, zorder=2, edgecolors='none')

        # PINN mean
        ax.plot(x, mean, '-', color=color, marker='D', markersize=5,
                linewidth=1.5, label='PINN mean', zorder=4)

        if cfg['logx']:
            ax.set_xscale('log')
        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel('Effective coefficient (GPa)')
        ax.set_title(LABELS[key])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)

    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f'Individual runs — {cfg["title_tag"]}',
        fontsize=13, y=1.02)
    plt.tight_layout()

    tag = '_'.join(keys)
    fname = f"runs_{tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved → {os.path.join(output_dir, fname + '.pdf')}")


# ── Figure: convergence history per run ──────────────────────────────────────

def _load_run_histories(cfg, data, key):
    """
    For a diagonal coefficient (p=1..6), load reff_history from each
    run's results.json.  Returns a list of dicts, one per parameter point:
        [{'param_value': ..., 'exact': ..., 'runs': [
            {'seed': ..., 'history': [...], 'converged': bool}, ...]}]
    """
    p_idx = KEY_TO_PROBLEM.get(key)
    if p_idx is None:
        return []  # cross-term — no per-cycle history

    base = cfg['base_dir']
    results = []
    for entry in data:
        pv = entry[cfg['param_col']]
        exact = entry['stats'][key]['exact'] if key in entry['stats'] else None
        folder = os.path.join(base, cfg['param_fmt'].format(pv))

        run_dirs = sorted(glob.glob(os.path.join(folder, 'run_*')))
        runs = []
        for rd in run_dirs:
            rfile = os.path.join(rd, 'results.json')
            if not os.path.isfile(rfile):
                continue
            with open(rfile, 'r') as f:
                rd_data = json.load(f)
            prob = next((pr for pr in rd_data['problems'] if pr['p'] == p_idx), None)
            if prob is None:
                continue
            runs.append({
                'seed': rd_data['seed'],
                'history': prob['reff_history'],
                'converged': prob['converged'],
                'final': prob['reff_final'],
            })
        results.append({'param_value': pv, 'exact': exact, 'runs': runs})
    return results


def fig_convergence_history(data, cfg, keys, output_dir):
    """
    For each coefficient and each parameter point, plot the prediction
    at each training cycle for every run, together with the AHM exact value.
    One subplot per parameter value; one figure per coefficient.

    The y-axis is auto-scaled to the "converged region" (cycles ≥ 2 plus
    the exact value), so that early wild transients don't flatten the plot.
    """
    cmap = plt.cm.tab10

    for key in keys:
        if key not in KEY_TO_PROBLEM:
            print(f"  Skipping {LABELS.get(key, key)}: cross-term has no "
                  f"per-cycle history.")
            continue

        points = _load_run_histories(cfg, data, key)
        if not points:
            continue

        n_pts = len(points)
        ncols = min(n_pts, 4)
        nrows = (n_pts + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.8 * nrows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for idx, pt in enumerate(points):
            ax = axes_flat[idx]
            exact = pt['exact']
            pv = pt['param_value']

            # ── Collect all values from cycle ≥ 2 to determine y-limits ──
            stable_vals = []
            for run in pt['runs']:
                h = run['history']
                stable_vals.extend(h[1:])       # skip cycle 1
            if exact is not None:
                stable_vals.append(exact)
            # Fallback: if fewer than 2 stable values, use everything
            if len(stable_vals) < 2:
                stable_vals = [v for run in pt['runs'] for v in run['history']]
                if exact is not None:
                    stable_vals.append(exact)

            if stable_vals:
                y_lo = min(stable_vals)
                y_hi = max(stable_vals)
                span = y_hi - y_lo if y_hi > y_lo else abs(y_hi) * 0.05 + 1e-6
                margin = 0.25 * span
                ylim_lo = y_lo - margin
                ylim_hi = y_hi + margin
            else:
                ylim_lo, ylim_hi = None, None

            for ri, run in enumerate(pt['runs']):
                h = run['history']
                color = cmap(ri % 10)
                label = f"seed {run['seed']}" if idx == 0 else None
                ax.plot(range(1, len(h) + 1), h, '-', color=color,
                        linewidth=0.9, alpha=0.7, label=label)
                # Mark final value
                ax.plot(len(h), run['final'], 'o', color=color,
                        markersize=4, zorder=5)

            # AHM exact
            if exact is not None:
                ax.axhline(exact, color='black', linestyle='--',
                           linewidth=1.3, label='AHM exact' if idx == 0 else None)

            # ── Apply the smart y-limits ──
            if ylim_lo is not None:
                ax.set_ylim(ylim_lo, ylim_hi)

            ax.set_xlabel('Cycle')
            ax.set_ylabel(f'{LABELS[key]} (GPa)')
            ax.set_title(f'{cfg["xlabel"]} = {pv:.4g}', fontsize=10)
            ax.grid(True, linestyle=':', linewidth=0.3, alpha=0.5)

        # Hide extra axes
        for j in range(n_pts, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Shared legend from first subplot
        handles, labs = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labs, loc='lower center',
                   ncol=min(len(handles), 6), fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f'{LABELS[key]} — prediction across training cycles\n'
            f'({cfg["title_tag"]})',
            fontsize=13, y=1.02)
        plt.tight_layout()

        fname = f"history_{key}"
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"), dpi=300)
        plt.close(fig)
        print(f"  Saved → {os.path.join(output_dir, fname + '.pdf')}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from saved parametric-study data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--study', choices=['contrast', 'vf'], required=True,
                        help='Which parametric study to read.')
    parser.add_argument('--coeff', type=str, default='all',
                        help='Comma-separated coefficient keys, or "all". '
                             'E.g.: R33_eff  or  R11_eff,R22_eff,R33_eff')
    parser.add_argument('--boxplot', action='store_true',
                        help='Also generate box-plot figure.')
    parser.add_argument('--scatter', action='store_true',
                        help='Overlay individual run scatter on absolute plot.')
    parser.add_argument('--runs', action='store_true',
                        help='Generate individual-runs scatter figure.')
    parser.add_argument('--history', action='store_true',
                        help='Plot prediction vs exact across training cycles '
                             'for each run (only diagonal coefficients).')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: <study_dir>/figures_diag/)')

    args = parser.parse_args()

    # Parse coefficient list
    if args.coeff.lower() == 'all':
        keys = ALL_KEYS
    else:
        keys = [k.strip() for k in args.coeff.split(',')]
        for k in keys:
            if k not in ALL_KEYS:
                sys.exit(f"ERROR: unknown coefficient '{k}'. "
                         f"Choose from: {', '.join(ALL_KEYS)}")

    data, cfg = load_data(args.study)

    out = args.output_dir
    if out is None:
        base = os.path.dirname(cfg['json'])
        out = os.path.join(os.path.dirname(base), 'figures_diag')
    os.makedirs(out, exist_ok=True)

    print(f"Study: {args.study}  |  Coefficients: {keys}")
    print(f"Output: {out}\n")

    fig_absolute(data, cfg, keys, out, scatter=args.scatter)
    fig_relerr(data, cfg, keys, out)

    if args.boxplot:
        fig_boxplot(data, cfg, keys, out)

    if args.runs:
        fig_individual_runs(data, cfg, keys, out)

    if args.history:
        fig_convergence_history(data, cfg, keys, out)

    print(f"\nDone. Figures in: {out}")


if __name__ == '__main__':
    main()
