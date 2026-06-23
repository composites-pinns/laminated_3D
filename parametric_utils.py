#!/usr/bin/env python3
"""
parametric_utils.py
===================
Shared utilities for parametric PINN homogenisation studies.

Provides:
  - Material stiffness matrix computation (isotropic)
  - Analytical (AHM) effective-coefficient formulas
  - Custom training-data and evaluation-point generation
  - A self-contained training loop (Adam + L-BFGS cycles with convergence check)
  - Single-parameter-point multi-run driver with resume capability
  - Result persistence (CSV / JSON)
"""

import os
import glob
import shutil
import time
import json
import math
import random
import statistics
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

from pinn.device import get_device
from pinn.config import (
    LAYERS, EPOCH_ADAMS, EPOCH_LBFGS, ADAM_LR, N_COLLOCATION,
    DTYPE, TRAIN_SEED,
    EPOCH_ADAMS_FOLLOWING_CYCLES, EPOCH_LBFGS_FOLLOWING_CYCLES,
    ADAM_LATERCYCLES_PRINT_EVERY, LBFGS_LATERCYCLES_PRINT_EVERY,
    ADAM_PRINT_EVERY, N_UNIQUENESS, N_POINTS,
)
from pinn.model import PINN

# ══════════════════════════════════════════════════════════════════════════════
# ANSI helpers
# ══════════════════════════════════════════════════════════════════════════════
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"
MAGENTA = "\033[35m"; CYAN = "\033[36m"

# Convergence defaults (matching pinn_final.py)
DEFAULT_TOLERANCE = 0.05   # %
MIN_CYCLES_BEFORE_CHECK = 10
CONVERGENCE_WINDOW = 5


# ══════════════════════════════════════════════════════════════════════════════
# Material computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_stiffness_matrix(E: float, nu: float) -> np.ndarray:
    """Return the 6×6 Voigt stiffness matrix for an isotropic material."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


# ══════════════════════════════════════════════════════════════════════════════
# Exact (AHM) effective properties
# ══════════════════════════════════════════════════════════════════════════════

def compute_exact_diagonal(VF1: float, VF2: float,
                           R1: np.ndarray, R2: np.ndarray) -> List[float]:
    """Six diagonal effective coefficients [R11, R22, R33, R44, R55, R66]."""
    def prom(c1, c2):
        return VF1 * c1 + VF2 * c2

    R11_1, R13_1, R33_1 = R1[0, 0], R1[0, 2], R1[2, 2]
    R22_1, R23_1 = R1[1, 1], R1[1, 2]
    R44_1, R55_1, R66_1 = R1[3, 3], R1[4, 4], R1[5, 5]

    R11_2, R13_2, R33_2 = R2[0, 0], R2[0, 2], R2[2, 2]
    R22_2, R23_2 = R2[1, 1], R2[1, 2]
    R44_2, R55_2, R66_2 = R2[3, 3], R2[4, 4], R2[5, 5]

    R11_eff = (prom(R11_1, R11_2)
               - prom(R13_1**2 / R33_1, R13_2**2 / R33_2)
               + prom(R13_1 / R33_1, R13_2 / R33_2)**2
               / prom(1 / R33_1, 1 / R33_2))

    R22_eff = (prom(R22_1, R22_2)
               - prom(R23_1**2 / R33_1, R23_2**2 / R33_2)
               + prom(R23_1 / R33_1, R23_2 / R33_2)**2
               / prom(1 / R33_1, 1 / R33_2))

    R33_eff = 1.0 / prom(1 / R33_1, 1 / R33_2)
    R44_eff = 1.0 / prom(1 / R44_1, 1 / R44_2)
    R55_eff = 1.0 / prom(1 / R55_1, 1 / R55_2)
    R66_eff = prom(R66_1, R66_2)

    return [R11_eff, R22_eff, R33_eff, R44_eff, R55_eff, R66_eff]


def compute_exact_cross(VF1: float, VF2: float,
                        R1: np.ndarray, R2: np.ndarray) -> Dict[str, float]:
    """Three cross-term effective coefficients {R12, R13, R23}."""
    def prom(c1, c2):
        return VF1 * c1 + VF2 * c2

    R12_1, R13_1, R23_1, R33_1 = R1[0, 1], R1[0, 2], R1[1, 2], R1[2, 2]
    R12_2, R13_2, R23_2, R33_2 = R2[0, 1], R2[0, 2], R2[1, 2], R2[2, 2]

    inv_R33 = prom(1 / R33_1, 1 / R33_2)
    R13_R33 = prom(R13_1 / R33_1, R13_2 / R33_2)
    R23_R33 = prom(R23_1 / R33_1, R23_2 / R33_2)

    R13_eff = R13_R33 / inv_R33
    R23_eff = R23_R33 / inv_R33
    R12_eff = (prom(R12_1, R12_2)
               - prom(R13_1 * R23_1 / R33_1, R13_2 * R23_2 / R33_2)
               + R13_R33 * R23_R33 / inv_R33)

    return {'R12_eff': R12_eff, 'R13_eff': R13_eff, 'R23_eff': R23_eff}


def compute_all_exact(VF1: float, VF2: float,
                      R1: np.ndarray, R2: np.ndarray) -> Dict[str, float]:
    """All nine effective coefficients as a dict."""
    diag = compute_exact_diagonal(VF1, VF2, R1, R2)
    cross = compute_exact_cross(VF1, VF2, R1, R2)
    return {
        'R11_eff': diag[0], 'R22_eff': diag[1], 'R33_eff': diag[2],
        'R44_eff': diag[3], 'R55_eff': diag[4], 'R66_eff': diag[5],
        **cross,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic environment
# ══════════════════════════════════════════════════════════════════════════════

def setup_deterministic(seed: int) -> torch.device:
    """Configure full determinism and return the compute device."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    if device.type == 'cuda':
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    torch.set_default_dtype(DTYPE)

    if hasattr(torch.backends, 'cuda'):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    return device


# ══════════════════════════════════════════════════════════════════════════════
# Data generation (parameterised by y_I)
# ══════════════════════════════════════════════════════════════════════════════

def build_train_data_custom(y_I: float, seed: int = TRAIN_SEED,
                            n_coll: int = N_COLLOCATION,
                            n_uniq: int = N_UNIQUENESS):
    """Build collocation, interface and uniqueness data for a custom y_I."""
    np.random.seed(seed)
    y_coll1 = np.random.uniform(0, y_I, n_coll)[:, None]
    y_coll2 = np.random.uniform(y_I, 1, n_coll)[:, None]
    y_inter = np.array([[y_I]])
    y_uniq1 = np.linspace(0, y_I, n_uniq)[:, None]
    y_uniq2 = np.linspace(y_I, 1, n_uniq)[:, None]
    return y_coll1, y_coll2, y_inter, y_uniq1, y_uniq2


def build_eval_points_custom(y_I: float, n_pts: int = N_POINTS):
    """Evaluation points in both material sub-domains."""
    y1 = np.linspace(0, y_I, n_pts)[:, None]
    y2 = np.linspace(y_I, 1, n_pts)[:, None]
    return y1, y2


# ══════════════════════════════════════════════════════════════════════════════
# Model evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model_custom(model: PINN, p: int, device: torch.device,
                          y_I: float,
                          exacts_diagonal: List[float]):
    """Evaluate predictions and relative errors (custom y_I and exacts)."""
    y1, y2 = build_eval_points_custom(y_I)
    eff_tuple = model.Effective_Properties(y1, y2, device)
    preds = [float(x) for x in eff_tuple]
    rel_errs = [
        abs(preds[i] - exacts_diagonal[i]) / abs(exacts_diagonal[i]) * 100.0
        if abs(exacts_diagonal[i]) > 1e-15 else float('nan')
        for i in range(6)
    ]
    return preds, exacts_diagonal, rel_errs


def extract_cross_terms_custom(models: Dict[int, PINN],
                               y_I: float,
                               device: torch.device) -> Dict[str, float]:
    """Extract R12, R13, R23 from the p=2 and p=3 models."""
    y1, y2 = build_eval_points_custom(y_I)
    cross: Dict[str, float] = {}
    if 2 in models:
        props = models[2].Effective_Properties(y1, y2, device)
        cross['R12_eff'] = float(props[0])
    if 3 in models:
        props = models[3].Effective_Properties(y1, y2, device)
        cross['R13_eff'] = float(props[0])
        cross['R23_eff'] = float(props[1])
    return cross


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_single_cycle(model: PINN, cycle_num: int,
                      n_adam: int, n_lbfgs: int,
                      quiet: bool = True,
                      epoch_offset: int = 0,
                      global_start_time: Optional[float] = None) -> int:
    """Execute one Adam + L-BFGS cycle and return the epoch count."""
    if global_start_time is None:
        global_start_time = time.perf_counter()

    if cycle_num > 0:
        try:
            model.reset_optimizers(adam=True, lbfgs=True)
            last_lr = getattr(model, '_last_adam_lr', None)
            if last_lr is not None:
                for pg in model.optimizer_Adam.param_groups:
                    pg['lr'] = float(last_lr)
        except Exception:
            pass

    if cycle_num > 0:
        cur_adam = int(EPOCH_ADAMS_FOLLOWING_CYCLES)
        cur_lbfgs = int(EPOCH_LBFGS_FOLLOWING_CYCLES)
        try:
            model.adam_print_every = max(1, ADAM_LATERCYCLES_PRINT_EVERY)
            model.lbfgs_print_every = max(1, LBFGS_LATERCYCLES_PRINT_EVERY)
        except Exception:
            pass
    else:
        cur_adam = int(n_adam)
        cur_lbfgs = int(n_lbfgs)
        log_every = max(1, ADAM_PRINT_EVERY)
        try:
            if hasattr(model, 'adam_print_every'):
                delattr(model, 'adam_print_every')
            if hasattr(model, 'lbfgs_print_every'):
                delattr(model, 'lbfgs_print_every')
        except Exception:
            pass

    log_every_val = max(1, ADAM_LATERCYCLES_PRINT_EVERY) if cycle_num > 0 else max(1, ADAM_PRINT_EVERY)

    model.train(
        nIter_Adam=cur_adam,
        nIter_LBFGS=cur_lbfgs,
        log_every=log_every_val,
        progress=False,
        quiet=quiet,
        epoch_offset=epoch_offset,
        start_time_override=global_start_time,
    )
    return cur_adam + cur_lbfgs


def _is_monotonic(values: List[float]) -> bool:
    if len(values) < 2:
        return True
    inc = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    dec = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return inc or dec


def _compute_final_reff(reff_history: List[float]):
    if len(reff_history) < CONVERGENCE_WINDOW:
        return (reff_history[-1] if reff_history else float('nan')), 'last'
    last = reff_history[-CONVERGENCE_WINDOW:]
    if _is_monotonic(last):
        return last[-1], 'last'
    return sum(last) / len(last), 'mean'


def train_problem_custom(p: int, device: torch.device,
                         param: dict, y_I: float,
                         exacts_diagonal: List[float],
                         tolerance: float = DEFAULT_TOLERANCE,
                         max_cycles: int = 100,
                         quiet: bool = True) -> Dict[str, Any]:
    """
    Train one PINN problem (p=1..6) with custom material parameters.

    Returns a dict with reff_final, convergence info, and the trained model.
    """
    # Build data with the current y_I
    y_coll1, y_coll2, y_inter, y_uniq1, y_uniq2 = build_train_data_custom(y_I)

    # Create model
    model = PINN(y_coll1, y_coll2, y_inter, y_uniq1, y_uniq2,
                 param=param, layers=LAYERS, device=device, prob_ind=p)

    for g in model.optimizer_Adam.param_groups:
        g['lr'] = ADAM_LR
    model._last_adam_lr = ADAM_LR

    reff_exact = exacts_diagonal[p - 1]
    reff_history: List[float] = []
    reff_means: List[float] = []
    epoch_offset = 0
    global_start = time.perf_counter()
    converged = False

    for cycle in range(max_cycles):
        cycle_epochs = _run_single_cycle(
            model, cycle, EPOCH_ADAMS, EPOCH_LBFGS,
            quiet=quiet, epoch_offset=epoch_offset,
            global_start_time=global_start,
        )
        epoch_offset += cycle_epochs

        preds, _, rel_errs = evaluate_model_custom(
            model, p, device, y_I, exacts_diagonal)
        reff_pred = preds[p - 1]
        reff_history.append(reff_pred)

        # Simple convergence criterion (same as pinn_final.py)
        n_cycles = cycle + 1
        if n_cycles >= CONVERGENCE_WINDOW:
            current_mean = sum(reff_history[-CONVERGENCE_WINDOW:]) / CONVERGENCE_WINDOW
            reff_means.append(current_mean)

            if (n_cycles >= MIN_CYCLES_BEFORE_CHECK
                    and len(reff_means) >= CONVERGENCE_WINDOW):
                mean_old = reff_means[-CONVERGENCE_WINDOW]
                mean_new = reff_means[-1]
                if abs(reff_exact) > 1e-12:
                    diff_pct = abs(mean_new - mean_old) / abs(reff_exact) * 100.0
                else:
                    diff_pct = abs(mean_new - mean_old) * 100.0
                if diff_pct < tolerance:
                    converged = True
                    break

    reff_final, reff_method = _compute_final_reff(reff_history)
    training_time = time.perf_counter() - global_start

    return {
        'p': p,
        'reff_final': float(reff_final),
        'reff_exact': float(reff_exact),
        'reff_method': reff_method,
        'reff_history': [float(v) for v in reff_history],
        'cycles': len(reff_history),
        'converged': converged,
        'total_epochs': epoch_offset,
        'time_sec': training_time,
        'model': model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Multi-run driver for a single parameter point
# ══════════════════════════════════════════════════════════════════════════════

def run_parameter_point(
    param_label: str,
    param_value: float,
    VF1: float,
    R_mat1: np.ndarray,
    R_mat2: np.ndarray,
    n_runs: int = 10,
    base_seed: int = TRAIN_SEED,
    targets: Optional[List[int]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_cycles: int = 100,
    quiet: bool = True,
    base_dir: str = '.',
) -> Dict[str, Any]:
    """
    Execute *n_runs* independent trainings for one parameter point.

    Supports resume: completed runs (with results.json) are loaded from disk.

    Returns dict with all_runs, stats, and exact values.
    """
    if targets is None:
        targets = [1, 2, 3, 4, 5, 6]

    VF2 = 1.0 - VF1
    y_I = VF1
    param = {'VF_mat1': VF1, 'VF_mat2': VF2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}
    exacts_diag = compute_exact_diagonal(VF1, VF2, R_mat1, R_mat2)
    exacts_cross = compute_exact_cross(VF1, VF2, R_mat1, R_mat2)
    exact_all = {
        **{f'R{p}{p}_eff': exacts_diag[p - 1] for p in range(1, 7)},
        **exacts_cross,
    }

    point_dir = os.path.join(base_dir, param_label)
    os.makedirs(point_dir, exist_ok=True)

    seeds = [base_seed + i * 1000 for i in range(n_runs)]
    all_runs: List[Dict] = []

    for run_idx, seed in enumerate(seeds, 1):
        run_dir = os.path.join(point_dir, f"run_{run_idx:02d}_seed_{seed}")
        result_file = os.path.join(run_dir, "results.json")

        # Resume: load cached results if available
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    run_data = json.load(f)
                all_runs.append(run_data)
                print(f"      Run {run_idx:>2}/{n_runs} (seed={seed}): "
                      f"{GREEN}loaded from cache{RESET}")
                continue
            except Exception:
                pass  # re-train if file is corrupt

        os.makedirs(run_dir, exist_ok=True)
        print(f"      Run {run_idx:>2}/{n_runs} (seed={seed}): training …")

        device = setup_deterministic(seed)

        reff_values: Dict[str, float] = {}
        models: Dict[int, PINN] = {}
        run_details: List[Dict] = []

        for p in sorted(targets):
            result = train_problem_custom(
                p, device, param, y_I, exacts_diag,
                tolerance=tolerance, max_cycles=max_cycles, quiet=quiet,
            )
            models[p] = result['model']
            reff_values[f'R{p}{p}_eff'] = result['reff_final']
            run_details.append({
                'p': result['p'],
                'reff_final': result['reff_final'],
                'reff_exact': result['reff_exact'],
                'cycles': result['cycles'],
                'converged': result['converged'],
                'time_sec': result['time_sec'],
                'reff_history': result['reff_history'],
            })

        # Cross terms
        cross = extract_cross_terms_custom(models, y_I, device)
        reff_values.update(cross)

        run_data = {
            'run': run_idx,
            'seed': seed,
            'param_label': param_label,
            'param_value': float(param_value),
            'VF1': float(VF1),
            'reff_values': {k: float(v) for k, v in reff_values.items()},
            'problems': run_details,
        }

        with open(result_file, 'w') as f:
            json.dump(run_data, f, indent=2)

        all_runs.append(run_data)

        # Free GPU/CPU memory
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Aggregate statistics ──
    prop_keys = ['R11_eff', 'R22_eff', 'R33_eff', 'R44_eff', 'R55_eff', 'R66_eff',
                 'R12_eff', 'R13_eff', 'R23_eff']

    all_reff: Dict[str, List[float]] = {}
    for key in prop_keys:
        vals = [
            run['reff_values'][key]
            for run in all_runs
            if key in run.get('reff_values', {})
        ]
        if vals:
            all_reff[key] = vals

    stats: Dict[str, Dict] = {}
    for key, vals in all_reff.items():
        exact_val = exact_all.get(key, float('nan'))
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        med = statistics.median(vals)
        stats[key] = {
            'mean': m,
            'std': s,
            'median': med,
            'min': min(vals),
            'max': max(vals),
            'exact': exact_val,
            'rel_err_mean_pct': (abs(m - exact_val) / abs(exact_val) * 100.0
                                 if abs(exact_val) > 1e-15 else float('nan')),
            'n': len(vals),
            'values': vals,
        }

    return {
        'param_label': param_label,
        'param_value': float(param_value),
        'VF1': float(VF1),
        'all_runs': all_runs,
        'all_reff': all_reff,
        'stats': stats,
        'exact': exact_all,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_sweep_csv(all_results: List[Dict], output_path: str,
                   param_col_name: str = 'parameter') -> str:
    """
    Write a comprehensive CSV with per-run and per-coefficient data.

    Columns: parameter, run, seed, R11_eff, …, R23_eff,
             R11_exact, …, R23_exact
    """
    prop_keys = ['R11_eff', 'R22_eff', 'R33_eff',
                 'R44_eff', 'R55_eff', 'R66_eff',
                 'R12_eff', 'R13_eff', 'R23_eff']

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        exact_cols = [k.replace('_eff', '_exact') for k in prop_keys]
        header = [param_col_name, 'run', 'seed'] + prop_keys + exact_cols
        f.write(','.join(header) + '\n')

        for res in all_results:
            exact = res['exact']
            for run in res['all_runs']:
                vals = run.get('reff_values', {})
                row = [
                    f"{res['param_value']:.6f}",
                    str(run['run']),
                    str(run['seed']),
                ]
                for k in prop_keys:
                    row.append(f"{vals[k]:.10f}" if k in vals else '')
                for k in prop_keys:
                    row.append(f"{exact[k]:.10f}" if k in exact else '')
                f.write(','.join(row) + '\n')

    print(f"  {MAGENTA}CSV saved → {output_path}{RESET}")
    return output_path


def save_sweep_statistics_csv(all_results: List[Dict], output_path: str,
                              param_col_name: str = 'parameter') -> str:
    """Write a CSV with mean / std / exact / error per parameter value."""
    prop_keys = ['R11_eff', 'R22_eff', 'R33_eff',
                 'R44_eff', 'R55_eff', 'R66_eff',
                 'R12_eff', 'R13_eff', 'R23_eff']

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        header = [param_col_name, 'property', 'n', 'mean', 'std', 'median',
                  'min', 'max', 'exact', 'rel_err_mean_pct']
        f.write(','.join(header) + '\n')

        for res in all_results:
            pv = res['param_value']
            for k in prop_keys:
                if k not in res['stats']:
                    continue
                s = res['stats'][k]
                row = [
                    f"{pv:.6f}", k, str(s['n']),
                    f"{s['mean']:.10f}", f"{s['std']:.10f}",
                    f"{s['median']:.10f}", f"{s['min']:.10f}",
                    f"{s['max']:.10f}", f"{s['exact']:.10f}",
                    f"{s['rel_err_mean_pct']:.6f}",
                ]
                f.write(','.join(row) + '\n')

    print(f"  {MAGENTA}Statistics CSV saved → {output_path}{RESET}")
    return output_path


def save_sweep_json(all_results: List[Dict], output_path: str,
                    param_col_name: str = 'parameter') -> str:
    """Write all results as JSON for flexible downstream analysis."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    serialisable = []
    for res in all_results:
        entry = {
            param_col_name: res['param_value'],
            'VF1': res['VF1'],
            'exact': res['exact'],
            'stats': {
                k: {kk: vv for kk, vv in v.items()}
                for k, v in res['stats'].items()
            },
        }
        serialisable.append(entry)

    with open(output_path, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"  {MAGENTA}JSON saved → {output_path}{RESET}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Formatting
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Sweep progress detection & resume prompt
# ══════════════════════════════════════════════════════════════════════════════

def detect_sweep_progress(
    base_dir: str,
    param_labels: List[str],
    n_runs: int,
    base_seed: int = TRAIN_SEED,
) -> Dict[str, Any]:
    """
    Scan *base_dir* for existing results.json files and classify each
    parameter point as 'complete', 'partial' or 'pending'.

    Returns::

        {
          'has_progress': bool,
          'points': [
              {'label': str, 'completed_runs': int, 'total_runs': int,
               'status': 'complete'|'partial'|'pending'},
              ...
          ],
          'n_complete': int,
          'n_partial': int,
          'n_pending': int,
          'last_complete_label': str | None,
        }
    """
    seeds = [base_seed + i * 1000 for i in range(n_runs)]
    points = []
    n_complete = n_partial = n_pending = 0
    last_complete_label = None

    for label in param_labels:
        point_dir = os.path.join(base_dir, label)
        done = 0
        for run_idx, seed in enumerate(seeds, 1):
            rf = os.path.join(point_dir,
                              f"run_{run_idx:02d}_seed_{seed}",
                              "results.json")
            if os.path.isfile(rf):
                done += 1

        if done == n_runs:
            status = 'complete'
            n_complete += 1
            last_complete_label = label
        elif done > 0:
            status = 'partial'
            n_partial += 1
            last_complete_label = label
        else:
            status = 'pending'
            n_pending += 1

        points.append({
            'label': label,
            'completed_runs': done,
            'total_runs': n_runs,
            'status': status,
        })

    return {
        'has_progress': (n_complete + n_partial) > 0,
        'points': points,
        'n_complete': n_complete,
        'n_partial': n_partial,
        'n_pending': n_pending,
        'last_complete_label': last_complete_label,
    }


def prompt_resume_or_restart(progress: Dict[str, Any],
                             base_dir: str) -> bool:
    """
    Print a summary of existing progress and ask the user whether to
    *continue* (True) or *restart from scratch* (False).

    If the user chooses to restart, all results under *base_dir* are deleted.
    """
    pts = progress['points']
    total = len(pts)
    nc, np_, npend = (progress['n_complete'],
                      progress['n_partial'],
                      progress['n_pending'])

    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Previous progress detected{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"  Complete : {nc}/{total} parameter points")
    if np_ > 0:
        print(f"  Partial  : {np_}/{total} parameter points")
    print(f"  Pending  : {npend}/{total} parameter points")
    print()

    # Show detail per point
    for p in pts:
        if p['status'] == 'complete':
            icon = f"{GREEN}✔{RESET}"
        elif p['status'] == 'partial':
            icon = f"{YELLOW}◐{RESET}"
        else:
            icon = f"  "
        runs_str = f"{p['completed_runs']}/{p['total_runs']} runs"
        print(f"    {icon} {p['label']:>20s}  {runs_str}")

    print()
    print(f"  {BOLD}[c]{RESET} Continue from where it was left off")
    print(f"  {BOLD}[r]{RESET} Restart from scratch (deletes existing results)")

    while True:
        try:
            choice = input(f"\n  Choice [c/r]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            choice = 'c'  # default to continue in non-interactive

        if choice in ('c', ''):
            print(f"  → Continuing from previous progress\n")
            return True
        if choice == 'r':
            # Confirm destructive action
            try:
                confirm = input(
                    f"  {YELLOW}This will delete all data in '{base_dir}'. "
                    f"Confirm? [y/N]: {RESET}"
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                confirm = 'n'
            if confirm == 'y':
                # Remove param-point directories only (keep data/ and figures/)
                for p in pts:
                    d = os.path.join(base_dir, p['label'])
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                # Also remove stale summary files
                for sub in ('data', 'figures'):
                    d = os.path.join(base_dir, sub)
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
                os.makedirs(os.path.join(base_dir, 'figures'), exist_ok=True)
                print(f"  → Starting fresh\n")
                return False
            else:
                print(f"  → Restart cancelled, continuing instead\n")
                return True
        print(f"  Please enter 'c' (continue) or 'r' (restart)")


def format_elapsed(sec: float) -> str:
    if sec is None:
        return "0s"
    sec = float(sec)
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}h{m:02d}m{int(s):02d}s"
    if sec >= 60:
        m = int(sec // 60)
        s = sec % 60
        return f"{m}m{int(s):02d}s"
    return f"{sec:.1f}s"
