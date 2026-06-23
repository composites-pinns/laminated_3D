#!/usr/bin/env python3
"""
Hyperparameter grid (sequential sweep) orchestrator for laminated PINNs.

Usage examples (PowerShell):

  # Default sweeps (quick minimal demo)
  & python hyper_grid.py --target 6 --quick

  # Custom sweep lists
  & python hyper_grid.py --target 6 --epochs-adams 400,800,1200 --epochs-lbfgs 30,50,80 --adam-lr 1e-3,5e-4,2e-4

Results:
  - Creates CSV files per parameter sweep in `grid_results/`.
  - Optionally creates PNG plots (requires matplotlib).
  - Color-coded progress logs.

Design:
  - Sequential sweeps: vary one hyperparameter at a time keeping others fixed at baseline from pinn.config.
  - Baseline values imported from pinn.config; direct overrides passed to training routine rather than modifying config globals.
  - By default trains only the target problem (`--target p`) using same network architecture to reduce cost.
  - You may add `--full-sequence` to run progressive training order for each configuration (costly).
"""

import os
import time
import csv
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import numpy as np

from pinn.device import get_device
from pinn.config import (
    LAYERS, EPOCH_ADAMS, EPOCH_LBFGS, ADAM_LR, N_COLLOCATION, N_UNIQUENESS, N_POINTS,
    VF_mat1, VF_mat2, DTYPE, SWEEP_CONFIG, EPOCH_ADAMS_FOLLOWING_CYCLES, EPOCH_LBFGS_FOLLOWING_CYCLES,
    ADAM_LATERCYCLES_PRINT_EVERY, LBFGS_LATERCYCLES_PRINT_EVERY,
    EPOCH_ADAMS_LAST, ADAM_PRINT_LAST_EVERY, ADAM_LR_LAST
)
from pinn import plot_utils
from pinn.materials import (
    R_mat1, R_mat2,
    R11_mat1, R12_mat1, R13_mat1, R22_mat1, R23_mat1, R33_mat1, R44_mat1, R55_mat1, R66_mat1,
    R11_mat2, R12_mat2, R13_mat2, R22_mat2, R23_mat2, R33_mat2, R44_mat2, R55_mat2, R66_mat2,
)
from pinn.data import build_train_data, build_eval_points
from pinn.model import PINN
import json
from multiprocessing import get_context

# ANSI colors/icons
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; GRAY = "\033[90m"
ICO_RUN = "🚀"; ICO_OK = "✅"; ICO_SKIP = "⏭️"; ICO_TIME = "⏱️"; ICO_SAVE = "💾"; ICO_GRID = "📊"; ICO_PARAM = "🧪"; ICO_WARN = "⚠️"

RESULTS_DIR = "grid_results"
RESULTS_DIR_DATA = os.path.join(RESULTS_DIR, f"data_{EPOCH_ADAMS}_epochs")
RESULTS_DIR_IMGS = os.path.join(RESULTS_DIR, f"images_{EPOCH_ADAMS}_epochs")
os.makedirs(RESULTS_DIR_DATA, exist_ok=True)
os.makedirs(RESULTS_DIR_IMGS, exist_ok=True)
RUNS_DONE_PATH = os.path.join(RESULTS_DIR_DATA, "runs_done.json")

GLOBAL_START = None
# How often (in Adam iterations) to print/log progress during Adam phases
ADAM_LOG_EVERY = 5000


def format_elapsed(sec: float) -> str:
    """Format elapsed seconds into HhMmSs or MmSs or Ss.s"""
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
    return f"{sec:.2f}s"


def load_runs_done() -> Dict[str, Any]:
    if os.path.exists(RUNS_DONE_PATH):
        try:
            with open(RUNS_DONE_PATH, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_run_done(key: str, entry: Dict[str, Any]):
    runs = load_runs_done()
    runs[key] = entry
    with open(RUNS_DONE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)


def save_run_file(key: str, entry: Dict[str, Any]):
    """Write per-run JSON file to avoid concurrent writes to runs_done.json."""
    # Try to parse key 'param|value|p<k>' and write JSON named
    # p{p}_{param}_{value}_run.json. Fall back to safe name if parsing fails.
    p_run, param, value = parse_key_for_p_param_value(key)
    try:
        if p_run is not None and param is not None:
            path = format_filename(p_run, param, value, 'run', dirpath=RESULTS_DIR_DATA, ext='.json')
        else:
            safe = _safe_name(str(key))
            path = os.path.join(RESULTS_DIR_DATA, f"run_{safe}.json")
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=2)
        return path
    except Exception:
        # fallback
        safe = _safe_name(str(key))
        path = os.path.join(RESULTS_DIR_DATA, f"run_{safe}.json")
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(entry, fh, ensure_ascii=False, indent=2)
            return path
        except Exception:
            return None


def _safe_name(key: str) -> str:
    return key.replace('/', '_').replace(' ', '_').replace('|', '_').replace(':', '_').replace('\\', '_').replace('/', '_')


def _model_path_for_key(key: str) -> str:
    # Try to parse keys of the form: <param>|<value>|p<k>
    parts = str(key).split('|') if key is not None else []
    try:
        if len(parts) >= 3 and parts[2].startswith('p') and parts[2][1:].isdigit():
            p_run = parts[2][1:]
            param = parts[0]
            value = parts[1]
            safe_val = str(value).replace('.', 'p').replace('e', 'e').replace('/', '_')
            fname = f"p{p_run}_{param}_{safe_val}_model.pt"
            return os.path.join(RESULTS_DIR_DATA, fname)
    except Exception:
        pass
    safe = _safe_name(key)
    return os.path.join(RESULTS_DIR_DATA, f"model_{safe}.pt")


def format_filename(p_run: int, param: str, value: Any, concept: str, dirpath: str = RESULTS_DIR_DATA, ext: str = '') -> str:
    """Return a path like: <dirpath>/p{p}_{param}_{safe_val}_{concept}{ext}

    `safe_val` replaces '.' -> 'p', '/' -> '_' etc to keep filenames safe.
    """
    safe_val = str(value).replace('.', 'p').replace('e', 'e').replace('/', '_')
    fname = f"{param}_{safe_val}_{concept}_p{p_run}{ext}"
    return os.path.join(dirpath, fname)


def _components_csv_for(loss_csv: str) -> str | None:
    """Return a CSV path that contains loss component columns.

    Prefer `loss_csv` if it already has component headers; otherwise try the
    companion `_run.csv` file which our orchestrator writes with component
    columns. Returns None if none found.
    """
    try:
        if not loss_csv:
            return None
        if os.path.exists(loss_csv):
            # read header only
            with open(loss_csv, 'r', newline='') as fh:
                hdr = fh.readline().strip()
            if any(h in hdr for h in ['loss_collocation', 'loss_uniqueness', 'loss_uniqueness_condition', 'loss_interface']):
                return loss_csv
        # Try companion run CSV
        run_csv = None
        if loss_csv.endswith('_loss.csv'):
            run_csv = loss_csv.replace('_loss.csv', '_run.csv')
        else:
            # fallback: replace last concept segment
            run_csv = loss_csv.replace('_loss', '_run')
        if run_csv and os.path.exists(run_csv):
            # inspect header
            with open(run_csv, 'r', newline='') as fh:
                hdr = fh.readline().strip()
            if any(h in hdr for h in ['loss_collocation', 'loss_uniqueness', 'loss_uniqueness_condition', 'loss_interface']):
                return run_csv
    except Exception:
        return None
    return None


def _write_loss_csv_with_weighted_components(loss_csv: str, model: PINN):
    """Write per-epoch loss CSV where component columns are multiplied by
    the model's current adaptive weights so their (weighted) sum equals
    the reported `loss_total` column.

    This is best-effort and will fall back to raw values if weights are
    inaccessible.
    """
    try:
        with open(loss_csv, 'w', newline='') as f:
            w = csv.writer(f)
            # Guardar tanto pérdidas crudas como contribuciones ponderadas (w*L)
            w.writerow(['epoch', 'loss_total', 'loss_collocation', 'loss_uniqueness', 'loss_uniqueness_condition', 'loss_interface',
                       'weighted_collocation', 'weighted_uniqueness', 'weighted_interface'])
            n_epochs = len(getattr(model, 'loss_log', []) or [])
            
            # Obtener logs de contribuciones ponderadas si existen
            weighted_coll = getattr(model, 'weighted_collocation_log', [])
            weighted_uniq = getattr(model, 'weighted_uniqueness_log', [])
            weighted_intf = getattr(model, 'weighted_interface_log', [])

            for idx in range(n_epochs):
                ep = idx + 1
                try:
                    lt = float(model.loss_log[idx])
                except Exception:
                    lt = float('nan')
                try:
                    lc = float(model.loss_collocation_log[idx])
                except Exception:
                    lc = float('nan')
                try:
                    lu_cond = float(model.loss_uniqueness_condition_log[idx])
                except Exception:
                    lu_cond = float('nan')
                try:
                    li = float(model.loss_interface_log[idx])
                except Exception:
                    li = float('nan')
                
                # Obtener contribuciones ponderadas reales (calculadas durante entrenamiento)
                try:
                    wc_contrib = float(weighted_coll[idx]) if idx < len(weighted_coll) else float('nan')
                except Exception:
                    wc_contrib = float('nan')
                try:
                    wu_contrib = float(weighted_uniq[idx]) if idx < len(weighted_uniq) else float('nan')
                except Exception:
                    wu_contrib = float('nan')
                try:
                    wi_contrib = float(weighted_intf[idx]) if idx < len(weighted_intf) else float('nan')
                except Exception:
                    wi_contrib = float('nan')

                w.writerow([ep, f"{lt:.12e}", f"{lc:.12e}", f"{lu_cond:.12e}", f"{lu_cond:.12e}", f"{li:.12e}",
                           f"{wc_contrib:.12e}", f"{wu_contrib:.12e}", f"{wi_contrib:.12e}"])
    except Exception:
        # best-effort: ignore failures
        pass


def parse_key_for_p_param_value(key: str):
    """Try to parse a key of the form 'param|value|p{n}' and return (p, param, value).
    Returns (None, None, None) if parsing fails.
    """
    if not key:
        return (None, None, None)
    parts = str(key).split('|')
    p = None; param = None; value = None
    try:
        # find p part
        for part in parts:
            if part.startswith('p') and part[1:].isdigit():
                p = int(part[1:])
                break
        if len(parts) >= 2:
            param = parts[0]
            value = parts[1]
        return (p, param, value)
    except Exception:
        return (None, None, None)


def get_p_from_key(key: str):
    parts = str(key).split('|') if key else []
    for part in parts:
        if part.startswith('p') and part[1:].isdigit():
            return int(part[1:])
    return None


def merge_run_files():
    runs = load_runs_done()
    # Scan the data directory for per-run JSONs (new format ends with '_run.json')
    for fname in os.listdir(RESULTS_DIR_DATA):
            if not (fname.endswith('_run.json') or (fname.startswith('model_') and fname.endswith('.pt'))):
                continue
            fpath = os.path.join(RESULTS_DIR_DATA, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    entry = json.load(fh)
                key = entry.get('key')
                if key:
                    runs[key] = entry.get('meta', entry)
            except Exception:
                continue
    with open(RUNS_DONE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)

@dataclass
class SweepResult:
    param: str
    value: Any
    trained_target: int
    time_sec: float
    final_loss: float
    rel_errs: List[float]
    preds: List[float]


def effective_exact_all() -> List[float]:
    def prom(CC1, CC2):
        return VF_mat1 * CC1 + VF_mat2 * CC2
    exacts: List[float] = []
    # p=1..6
    exacts.append(prom(R11_mat1, R11_mat2) - prom(R13_mat1**2 / R33_mat1, R13_mat2**2 / R33_mat2) + prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2) ** 2 / prom(1 / R33_mat1, 1 / R33_mat2))
    exacts.append(prom(R22_mat1, R22_mat2) - prom(R23_mat1**2 / R33_mat1, R23_mat2**2 / R33_mat2) + prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2) ** 2 * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1))
    exacts.append(prom(1 / R33_mat1, 1 / R33_mat2) ** (-1))
    exacts.append(prom(1 / R44_mat1, 1 / R44_mat2) ** (-1))
    exacts.append(prom(1 / R55_mat1, 1 / R55_mat2) ** (-1))
    exacts.append(prom(R66_mat1, R66_mat2))
    return exacts


def train_with_cycles(model: PINN, n_adam: int, n_lbfgs: int, log_every: int, progress: bool,
                      callback=None, callback_every: int = 10, cycles: int = 1):
    """Run model.train repeating the same (Adam, LBFGS) phases for `cycles`.

    If a callback is provided, epoch numbers are offset so they remain monotonic
    across cycles.  Elapsed time and printed iteration numbers are cumulative.
    """
    try:
        cycles = int(cycles)
    except Exception:
        cycles = 1
    if cycles < 1:
        cycles = 1
    epoch_offset = 0
    # Single start_time shared across all cycles so elapsed time is cumulative
    global_start_time = time.perf_counter()
    for c in range(cycles):
        if c > 0:
            try:
                model.reset_optimizers(adam=True, lbfgs=True)
                # reapply Adam LR saved on the model (important when recreating optimizer)
                last_lr = getattr(model, '_last_adam_lr', None)
                if last_lr is not None:
                    for pg in model.optimizer_Adam.param_groups:
                        pg['lr'] = float(last_lr)
            except Exception:
                pass
        # Choose Adam/LBFGS epochs for this cycle (later cycles may be shorter)
        if c > 0:
            current_n_adam = int(EPOCH_ADAMS_FOLLOWING_CYCLES)
            current_n_lbfgs = int(EPOCH_LBFGS_FOLLOWING_CYCLES)
        else:
            current_n_adam = int(n_adam)
            current_n_lbfgs = int(n_lbfgs)
        # Configure logging cadence: use later-cycle print frequencies if c>0
        if c > 0:
            current_log_every = int(ADAM_LATERCYCLES_PRINT_EVERY) if current_n_adam > 0 else 1
            # set attributes on model so train() uses them for elapsed-time prints
            try:
                model.adam_print_every = int(ADAM_LATERCYCLES_PRINT_EVERY)
            except Exception:
                model.adam_print_every = None
            try:
                model.lbfgs_print_every = int(LBFGS_LATERCYCLES_PRINT_EVERY)
            except Exception:
                model.lbfgs_print_every = None
        else:
            current_log_every = log_every
            # restore defaults (or remove overrides)
            try:
                if hasattr(model, 'adam_print_every'):
                    delattr(model, 'adam_print_every')
            except Exception:
                pass
            try:
                if hasattr(model, 'lbfgs_print_every'):
                    delattr(model, 'lbfgs_print_every')
            except Exception:
                pass
        cb = None
        if callback is not None:
            def _cb(m, ep, _off=epoch_offset):
                return callback(m, ep + _off)
            cb = _cb
        model.train(
            nIter_Adam=current_n_adam,
            nIter_LBFGS=current_n_lbfgs,
            log_every=current_log_every,
            progress=progress,
            callback=cb,
            callback_every=callback_every,
            epoch_offset=epoch_offset,
            start_time_override=global_start_time,
        )
        epoch_offset += int(current_n_adam) + int(current_n_lbfgs)

    # Optional final Adam phase after all cycles
    try:
        if int(EPOCH_ADAMS_LAST) > 0:
            final_adam_epochs = int(EPOCH_ADAMS_LAST)
            final_log_every = int(ADAM_PRINT_LAST_EVERY) if (ADAM_PRINT_LAST_EVERY is not None) else 1
            print(f"{CYAN}Starting final Adam phase: {final_adam_epochs} epochs{RESET}")
            cb_final = None
            if callback is not None:
                def _cb_final(m, ep, _off=epoch_offset):
                    return callback(m, ep + _off)
                cb_final = _cb_final
            # Run only Adam (LBFGS iterations = 0)
            # Override Adam LR for this final phase if configured
            try:
                for pg in model.optimizer_Adam.param_groups:
                    pg['lr'] = float(ADAM_LR_LAST)
                try:
                    model._last_adam_lr = float(ADAM_LR_LAST)
                except Exception:
                    model._last_adam_lr = None
            except Exception:
                pass

            model.train(
                nIter_Adam=final_adam_epochs,
                nIter_LBFGS=0,
                log_every=final_log_every,
                progress=progress,
                callback=cb_final,
                callback_every=callback_every,
                epoch_offset=epoch_offset,
                start_time_override=global_start_time,
            )
            epoch_offset += final_adam_epochs
    except Exception:
        pass


def train_single_problem(p: int, device: torch.device, epochs_adams: int, epochs_lbfgs: int,
                         adam_lr: float, layers: List[int], relerr_callback=None, callback_every: int = 10,
                         sweep_param_name: str | None = None, sweep_param_value: Any = None,
                         cycles: int = 1) -> PINN:
    # Build data once (global for all runs) - we rely on external functions using config sizes
    y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2 = build_train_data()
    PARAM = {'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}
    model = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
                 param=PARAM, layers=layers, device=device, prob_ind=p)
    try:
        model.sweep_param_name = sweep_param_name
        model.sweep_param_value = sweep_param_value
    except Exception:
        pass
    # Override optimizers hyperparams (Adam LR, epochs) dynamically
    for g in model.optimizer_Adam.param_groups:
        g['lr'] = adam_lr
    # remember the intended Adam LR so we can reapply it if the optimizer is reset between cycles
    try:
        model._last_adam_lr = float(adam_lr)
    except Exception:
        model._last_adam_lr = None
    start = time.perf_counter()
    # Use fixed, less-frequent logging for Adam phases to reduce console spam
    train_with_cycles(model, epochs_adams, epochs_lbfgs, (ADAM_LOG_EVERY if epochs_adams > 0 else 1), False,
                      callback=relerr_callback, callback_every=callback_every, cycles=cycles)
    dt = time.perf_counter() - start
    return model, dt


def evaluate_model(model: PINN, p: int, device: torch.device) -> float:
    y_points1, y_points2 = build_eval_points()
    eff_tuple = model.Effective_Properties(y_points1, y_points2, device)
    preds = [float(x) for x in eff_tuple]
    exacts = effective_exact_all()
    rel_errs = [abs(preds[i] - exacts[i]) / abs(exacts[i]) * 100.0 if exacts[i] != 0 else float('nan') for i in range(6)]
    return preds, exacts, rel_errs


def _read_relerr_csv_and_plot(run_csv: str, p_run: int, param_name: str, value: Any, imgs_dir: str, cycles: int = 1):
    """Lee el CSV de relerr y genera gráficas de RelErr y R_eff evolution.
    
    Args:
        run_csv: Ruta al archivo CSV (con columnas epoch, rel_err, R_eff_pred, R_eff_exact, ...)
        p_run: Índice del problema (1-6)
        param_name: Nombre del parámetro del barrido
        value: Valor del parámetro
        imgs_dir: Directorio donde guardar las imágenes
    """
    if not os.path.exists(run_csv):
        return
    
    xs = []
    ys = []
    reff_preds = []
    reff_exact = None
    
    try:
        with open(run_csv, 'r', newline='') as fh:
            r = csv.reader(fh)
            headers = next(r)
            try:
                idx_epoch = headers.index('epoch')
                # Buscar columna de rel_err (puede variar el nombre)
                idx_rel = None
                for i, h in enumerate(headers):
                    if 'rel_err' in h.lower():
                        idx_rel = i
                        break
                if idx_rel is None:
                    idx_rel = 1  # fallback
            except Exception:
                idx_epoch = 0
                idx_rel = 1
            
            # Buscar columnas de R_eff si existen (pueden tener sufijo _p{n})
            idx_reff_pred = None
            idx_reff_exact = None
            for i, h in enumerate(headers):
                if h.startswith('R_eff_pred'):
                    idx_reff_pred = i
                if h.startswith('R_eff_exact'):
                    idx_reff_exact = i
            
            for row in r:
                if not row:
                    continue
                try:
                    ep = int(row[idx_epoch])
                except Exception:
                    continue
                rel = row[idx_rel].strip() if idx_rel < len(row) else ''
                if rel == '':
                    continue
                try:
                    relf = float(rel)
                except Exception:
                    continue
                xs.append(ep)
                ys.append(relf)
                
                # Extraer R_eff si disponible
                if idx_reff_pred is not None and idx_reff_pred < len(row):
                    try:
                        reff_preds.append(float(row[idx_reff_pred]))
                    except:
                        reff_preds.append(float('nan'))
                if idx_reff_exact is not None and reff_exact is None and idx_reff_exact < len(row):
                    try:
                        reff_exact = float(row[idx_reff_exact])
                    except:
                        pass
    except Exception:
        return
    
    if not xs or not ys:
        return
    
    # Graficar RelErr
    plot_path_rel = format_filename(p_run, param_name, value, 'relerr', dirpath=imgs_dir, ext='.png')
    try:
        plot_utils.plot_relerr(xs, ys, 'Epoch', f"RelErr p{p_run} {param_name}={value}", plot_path_rel, logx=False, smooth_alpha=None)
        print(f"{MAGENTA}{ICO_SAVE} Guardado plot: {plot_path_rel}{RESET}")
    except Exception:
        pass
    
    # Graficar RelErr suavizado
    try:
        plot_path_rel_s = format_filename(p_run, param_name, value, 'relerr_smooth', dirpath=imgs_dir, ext='.png')
        plot_utils.plot_relerr(xs, ys, 'Epoch', f"RelErr p{p_run} {param_name}={value}", plot_path_rel_s, logx=False, smooth_alpha=0.12)
        print(f"{MAGENTA}{ICO_SAVE} Guardado plot (smoothed): {plot_path_rel_s}{RESET}")
    except Exception:
        pass
    
    # Graficar evolución de R_eff si disponible
    if reff_preds and reff_exact is not None:
        try:
            plot_path_reff = format_filename(p_run, param_name, value, 'reff_evolution', dirpath=imgs_dir, ext='.png')
            # Compute transition epochs (Adam->LBFGS and LBFGS->Adam) based on configured epoch lengths
            try:
                trans = []
                max_epoch = max(xs) if xs else 0
                cur = 0
                for c in range(1, int(cycles) + 1):
                    if c == 1:
                        a = int(EPOCH_ADAMS)
                        b = int(EPOCH_LBFGS)
                    else:
                        a = int(EPOCH_ADAMS_FOLLOWING_CYCLES)
                        b = int(EPOCH_LBFGS_FOLLOWING_CYCLES)
                    # Adam -> LBFGS
                    cur += a
                    if cur <= max_epoch:
                        trans.append((cur, 'adam->lbfgs'))
                    # LBFGS -> Adam
                    cur += b
                    if cur <= max_epoch:
                        trans.append((cur, 'lbfgs->adam'))
                transitions = trans
            except Exception:
                transitions = []
            plot_utils.plot_reff_evolution(xs, reff_preds, reff_exact, p_run, plot_path_reff, smooth_alpha=None, transitions=transitions)
            print(f"{MAGENTA}{ICO_SAVE} Guardado plot: {plot_path_reff}{RESET}")
        except Exception as e:
            print(f"{YELLOW}Error dibujando R_eff evolution: {e}{RESET}")
        
        try:
            plot_path_reff_s = format_filename(p_run, param_name, value, 'reff_evolution_smooth', dirpath=imgs_dir, ext='.png')
            plot_utils.plot_reff_evolution(xs, reff_preds, reff_exact, p_run, plot_path_reff_s, smooth_alpha=0.12)
            print(f"{MAGENTA}{ICO_SAVE} Guardado plot (smoothed): {plot_path_reff_s}{RESET}")
        except Exception:
            pass


def _make_relerr_callback(param_name: str, value: Any, p_run: int, device: torch.device, data_dir: str = RESULTS_DIR_DATA, every: int = 10, force_new: bool = False):
    """Return a callback function that appends relative errors to a CSV every `every` epochs.

    The callback signature is callback(model, epoch_int).
    
    Args:
        force_new: If True, always create a fresh CSV file (useful for --force-retrain).
    """
    rel_csv = format_filename(p_run, param_name, value, 'relerr', dirpath=data_dir, ext='.csv')
    # Ensure header exists: epoch, rel_err_p{p_run}_pct, R_eff_pred, R_eff_exact, loss_total, loss_collocation, loss_uniqueness_condition, loss_interface
    # If force_new or file doesn't exist, create fresh file with correct headers
    if force_new or not os.path.exists(rel_csv):
        with open(rel_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            header = ['epoch', f'rel_err_p{p_run}_pct', f'R_eff_pred_p{p_run}', f'R_eff_exact_p{p_run}', 'loss_total', 'loss_collocation', 'loss_uniqueness_condition', 'loss_interface']
            w.writerow(header)

    def _cb(model: PINN, epoch: int):
        # Only record at requested frequency (caller should ensure this)
        try:
            preds, exacts, rel_errs = evaluate_model(model, p_run, device)
            # Obtain the rel err for this specific problem index (p_run is 1-based)
            try:
                rel_val = float(rel_errs[p_run - 1]) if rel_errs and len(rel_errs) >= p_run else float('nan')
                reff_pred = float(preds[p_run - 1]) if preds and len(preds) >= p_run else float('nan')
                reff_exact = float(exacts[p_run - 1]) if exacts and len(exacts) >= p_run else float('nan')
            except Exception:
                rel_val = float('nan')
                reff_pred = float('nan')
                reff_exact = float('nan')
            # Try to obtain loss components from the model (tuple: total, coll, uniq, inter)
            try:
                lt, lc, lu, li = model.loss()
                # detach floats
                lt_f = float(lt.detach().cpu().item())
                lc_f = float(lc.detach().cpu().item())
                lu_f = float(lu.detach().cpu().item())
                li_f = float(li.detach().cpu().item())
            except Exception:
                # Fallback to logs if available
                lt_f = float(model.loss_log[-1]) if getattr(model, 'loss_log', None) else float('nan')
                lc_f = float(model.loss_collocation_log[-1]) if getattr(model, 'loss_collocation_log', None) else float('nan')
                lu_f = float(model.loss_uniqueness_condition_log[-1]) if getattr(model, 'loss_uniqueness_condition_log', None) else float('nan')
                li_f = float(model.loss_interface_log[-1]) if getattr(model, 'loss_interface_log', None) else float('nan')
            # Multiply component values by adaptive weights (if present) so the
            # saved columns reflect weighted contributions that sum to loss_total.
            try:
                wc = float(model.w_collocation.detach().cpu().item())
            except Exception:
                wc = float('nan')
            try:
                wu = float(model.w_uniqueness.detach().cpu().item())
            except Exception:
                wu = float('nan')
            try:
                wi = float(model.w_interface.detach().cpu().item())
            except Exception:
                wi = float('nan')
            try:
                if not math.isnan(lc_f):
                    lc_f = lc_f * wc
            except Exception:
                pass
            try:
                if not math.isnan(lu_f):
                    lu_f = lu_f * wu
            except Exception:
                pass
            try:
                if not math.isnan(li_f):
                    li_f = li_f * wi
            except Exception:
                pass
            with open(rel_csv, 'a', newline='') as fh:
                w = csv.writer(fh)
                # write epoch, rel_err, R_eff_pred, R_eff_exact, then loss components
                w.writerow([epoch, f"{rel_val:.6f}", f"{reff_pred:.6f}", f"{reff_exact:.6f}", f"{lt_f:.12e}", f"{lc_f:.12e}", f"{lu_f:.12e}", f"{li_f:.12e}"])
        except Exception:
            # swallow exceptions to avoid breaking training; logging could be added
            pass

    return _cb


def write_run_csv(param_name: str, value: Any, p_run: int, model: PINN, total_time: float, data_dir: str = RESULTS_DIR_DATA):
    """Write a single CSV per parameter combination and problem with columns:
    epoch, loss, rel_err, tiempo

    - `loss` comes from `model.loss_log` (one entry per epoch).
    - `rel_err` is the mean relative error across the six effective coefficients
      if available from the per-run relerr CSV created by the callback; otherwise
      the field is left empty for that epoch.
    - `tiempo` is the total training time in seconds (repeated for every row).
    """
    run_csv = format_filename(p_run, param_name, value, 'run', dirpath=data_dir, ext='.csv')
    # Try to read per-epoch relerr CSV (if callback produced it)
    rel_csv = format_filename(p_run, param_name, value, 'relerr', dirpath=data_dir, ext='.csv')
    rel_map = {}
    # If relerr CSV exists, parse it dynamically by header names to support
    # per-problem relerr CSVs that contain only one rel_err column.
    if os.path.exists(rel_csv):
        try:
            with open(rel_csv, 'r', newline='') as fh:
                r = csv.reader(fh)
                headers = next(r)
                # find rel_err columns (those starting with 'rel_err_p')
                rel_indices = [i for i, h in enumerate(headers) if h.startswith('rel_err_p')]
                # find named loss components if present
                idx_loss_total = headers.index('loss_total') if 'loss_total' in headers else None
                idx_loss_coll = headers.index('loss_collocation') if 'loss_collocation' in headers else None
                idx_loss_uniq = headers.index('loss_uniqueness_condition') if 'loss_uniqueness_condition' in headers else None
                idx_loss_inter = headers.index('loss_interface') if 'loss_interface' in headers else None
                for row in r:
                    if not row:
                        continue
                    try:
                        ep = int(row[0])
                    except Exception:
                        continue
                    rel_vals = []
                    for i in rel_indices:
                        try:
                            rel_vals.append(float(row[i]))
                        except Exception:
                            rel_vals.append(float('nan'))
                    mean_rel = None
                    valid = [v for v in rel_vals if not math.isnan(v)]
                    if valid:
                        mean_rel = sum(valid) / len(valid)
                    # Optional loss components by header
                    loss_total = None; loss_coll = None; loss_uniq = None; loss_inter = None
                    try:
                        if idx_loss_total is not None and idx_loss_total < len(row):
                            loss_total = float(row[idx_loss_total])
                    except Exception:
                        loss_total = None
                    try:
                        if idx_loss_coll is not None and idx_loss_coll < len(row):
                            loss_coll = float(row[idx_loss_coll])
                    except Exception:
                        loss_coll = None
                    try:
                        if idx_loss_uniq is not None and idx_loss_uniq < len(row):
                            loss_uniq = float(row[idx_loss_uniq])
                    except Exception:
                        loss_uniq = None
                    try:
                        if idx_loss_inter is not None and idx_loss_inter < len(row):
                            loss_inter = float(row[idx_loss_inter])
                    except Exception:
                        loss_inter = None
                    rel_map[ep] = {'mean_rel': mean_rel, 'loss_total': loss_total, 'loss_collocation': loss_coll, 'loss_uniqueness_condition': loss_uniq, 'loss_interface': loss_inter}
        except Exception:
            rel_map = {}

    # Write unified run CSV
    try:
        with open(run_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            # epoch, loss_total, loss_collocation, loss_uniqueness_condition, loss_interface, rel_err_mean, tiempo
            w.writerow(['epoch', 'loss_total', 'loss_collocation', 'loss_uniqueness_condition', 'loss_interface', 'rel_err', 'tiempo'])
            # Use available per-epoch logs; fall back to model.loss_log and component logs
            n_epochs = len(model.loss_log or [])
            for idx in range(n_epochs):
                ep = idx + 1
                # Prefer values from rel_map (which may contain loss components recorded by callback)
                rinfo = rel_map.get(ep, {}) if isinstance(rel_map.get(ep, {}), dict) else {}
                rel = rinfo.get('mean_rel', None)
                lt = rinfo.get('loss_total', None)
                lc = rinfo.get('loss_collocation', None)
                lu = rinfo.get('loss_uniqueness_condition', None)
                li = rinfo.get('loss_interface', None)

                # Fallbacks
                if lt is None:
                    try:
                        lt = float(model.loss_log[idx])
                    except Exception:
                        lt = float('nan')
                if lc is None:
                    try:
                        # read raw log and multiply by adaptive weight if present
                        raw_lc = float(model.loss_collocation_log[idx])
                        try:
                            wc = float(model.w_collocation.detach().cpu().item())
                        except Exception:
                            wc = 1.0
                        lc = raw_lc * wc
                    except Exception:
                        lc = float('nan')
                if lu is None:
                    try:
                        raw_lu = float(model.loss_uniqueness_condition_log[idx])
                        try:
                            wu = float(model.w_uniqueness.detach().cpu().item())
                        except Exception:
                            wu = 1.0
                        lu = raw_lu * wu
                    except Exception:
                        lu = float('nan')
                if li is None:
                    try:
                        raw_li = float(model.loss_interface_log[idx])
                        try:
                            wi = float(model.w_interface.detach().cpu().item())
                        except Exception:
                            wi = 1.0
                        li = raw_li * wi
                    except Exception:
                        li = float('nan')

                rel_str = f"{rel:.6f}" if (rel is not None) else ""
                w.writerow([ep, f"{lt:.12e}", f"{lc:.12e}", f"{lu:.12e}", f"{li:.12e}", rel_str, f"{total_time:.6f}"])
    except Exception:
        # best-effort: ignore write failures
        pass


def _worker_run_task(args_tuple):
    """Worker function used by multiprocessing Pool.
    args_tuple: (task_dict, device_type_str)
    """
    task, device_type = args_tuple
    try:
        # Limit threads inside worker
        try:
            import torch as _torch
            _torch.set_num_threads(1)
        except Exception:
            pass
        # If n_collocation is being varied, set env before building data
        if task.get('param_name') == 'n_collocation':
            os.environ['PINN_N_COLLOCATION'] = str(task.get('value'))
        device = torch.device(device_type)
        # prepare relerr callback for this task (records into data dir every 10 epochs)
        # Pass force_new=True when force_retrain so that old CSV with missing columns is replaced
        try:
            relerr_cb = _make_relerr_callback(task.get('param_name', 'unknown'), task.get('value', 'v'), task.get('p_run', 1), device, data_dir=RESULTS_DIR_DATA, every=10, force_new=task.get('force_retrain', False))
        except Exception:
            relerr_cb = None
        # Announce start of this task (helps track per-value progress)
        try:
            print(f"{CYAN}Iniciando entrenamiento {task.get('param_name')}={task.get('value')} para p{task.get('p_run')}{RESET}")
        except Exception:
            pass
        # If a trained model already exists for this task and force_retrain not set, load it and skip training
        try:
            meta_key = f"{task['param_name']}|{task['value']}|p{task['p_run']}"
            model_path = _model_path_for_key(meta_key)
        except Exception:
            model_path = None
        if model_path and os.path.exists(model_path) and not task.get('force_retrain', False):
            try:
                # instantiate model and load state
                model = PINN(*build_train_data(), param={'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}, layers=task['layers'], device=device, prob_ind=task['p_run'])
                try:
                    model.sweep_param_name = task.get('param_name')
                    model.sweep_param_value = task.get('value')
                except Exception:
                    pass
                try:
                    model.sweep_param_name = task.get('param_name')
                    model.sweep_param_value = task.get('value')
                except Exception:
                    pass
                sd = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(sd)
                preds, exacts, rel_errs = evaluate_model(model, task['p_run'], device)
                final_loss = model.loss_log[-1] if getattr(model, 'loss_log', None) else float('nan')
                # write unified run CSV (epoch, loss, rel_err, tiempo)
                try:
                    write_run_csv(task['param_name'], task['value'], task['p_run'], model, 0.0, data_dir=RESULTS_DIR_DATA)
                except Exception:
                    pass
                # Also write a per-epoch loss CSV with components so plots can be regenerated
                try:
                    loss_csv = format_filename(task['p_run'], task['param_name'], task['value'], 'loss', dirpath=RESULTS_DIR_DATA, ext='.csv')
                    _write_loss_csv_with_weighted_components(loss_csv, model)
                except Exception:
                    loss_csv = None
                # Try to regenerate per-run plots as well (loss overlay, relerr, components)
                try:
                    if loss_csv and os.path.exists(loss_csv):
                        plot_path_loss = format_filename(task['p_run'], task['param_name'], task['value'], 'loss', dirpath=RESULTS_DIR_IMGS, ext='.png')
                        try:
                            plot_utils.plot_loss_overlays([loss_csv], [str(task['value'])], plot_path_loss, smooth_alpha=None)
                        except Exception:
                            pass
                        # smoothed loss plot
                        try:
                            plot_path_loss_s = format_filename(task['p_run'], task['param_name'], task['value'], 'loss_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                            plot_utils.plot_loss_overlays([loss_csv], [str(task['value'])], plot_path_loss_s, smooth_alpha=0.12)
                        except Exception:
                            pass
                        # components plot
                        try:
                            plot_path_components = format_filename(task['p_run'], task['param_name'], task['value'], 'loss_components', dirpath=RESULTS_DIR_IMGS, ext='.png')
                            comp_csv = _components_csv_for(loss_csv)
                            if comp_csv:
                                plot_utils.plot_loss_components(comp_csv, plot_path_components, smooth_alpha=None)
                        except Exception:
                            pass
                        # smoothed components plot
                        try:
                            plot_path_components_s = format_filename(task['p_run'], task['param_name'], task['value'], 'loss_components_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                            comp_csv = _components_csv_for(loss_csv)
                            if comp_csv:
                                plot_utils.plot_loss_components(comp_csv, plot_path_components_s, smooth_alpha=0.12)
                        except Exception:
                            pass
                    # Also regenerate relerr plot from the unified run CSV if present
                    try:
                        run_csv = format_filename(task['p_run'], task['param_name'], task['value'], 'run', dirpath=RESULTS_DIR_DATA, ext='.csv')
                        if os.path.exists(run_csv):
                            xs = []
                            ys = []
                            with open(run_csv, 'r', newline='') as fh:
                                r = csv.reader(fh)
                                headers = next(r)
                                try:
                                    idx_epoch = headers.index('epoch')
                                    idx_rel = headers.index('rel_err')
                                except Exception:
                                    idx_epoch = 0; idx_rel = 5
                                for row in r:
                                    if not row:
                                        continue
                                    try:
                                        ep = int(row[idx_epoch])
                                    except Exception:
                                        continue
                                    rel = row[idx_rel].strip()
                                    if rel == '':
                                        continue
                                    try:
                                        relf = float(rel)
                                    except Exception:
                                        continue
                                    xs.append(ep)
                                    ys.append(relf)
                            if xs and ys:
                                plot_path_rel = format_filename(task['p_run'], task['param_name'], task['value'], 'relerr', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                try:
                                    plot_utils.plot_relerr(xs, ys, 'Epoch', f"RelErr p{task['p_run']} {task['param_name']}={task['value']}", plot_path_rel, logx=False, smooth_alpha=None)
                                except Exception:
                                    pass
                                # smoothed relerr
                                try:
                                    plot_path_rel_s = format_filename(task['p_run'], task['param_name'], task['value'], 'relerr_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                    plot_utils.plot_relerr(xs, ys, 'Epoch', f"RelErr p{task['p_run']} {task['param_name']}={task['value']}", plot_path_rel_s, logx=False, smooth_alpha=0.12)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass
                meta = {'key': meta_key, 'time_sec': 0.0, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
                save_run_file(meta_key, {'key': meta_key, 'meta': meta})
                return {'param_name': task['param_name'], 'value': task['value'], 'p_run': task['p_run'], 'time_sec': 0.0, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
            except Exception:
                # fallthrough to regular training
                pass
        # If sweeping LBFGS epochs, ensure Adam phase is identical across runs by
        # creating (or reusing) a baseline model state after Adam, then running
        # LBFGS from that state. This avoids differences in the Adam phase.
        def _baseline_key_for_task(t):
            # include parameters that affect the Adam phase
            k = f"adam|p{t['p_run']}|lr{t['adam_lr']}|layers{'-'.join(map(str,t['layers']))}|ncoll{os.environ.get('PINN_N_COLLOCATION', '')}|adamepochs{t['epochs_adams']}|cycles{t.get('cycles', 1)}"
            return k

        def _baseline_path_for_key(key):
            # Name baseline files with p{p}_<safe>_adam_baseline.pt when possible
            try:
                pnum = get_p_from_key(key)
                safe = _safe_name(str(key))
                if pnum is not None:
                    return os.path.join(RESULTS_DIR_DATA, f"p{pnum}_{safe}_adam_baseline.pt")
                return os.path.join(RESULTS_DIR_DATA, f"adam_baseline_{safe}.pt")
            except Exception:
                safe = _safe_name(str(key))
                return os.path.join(RESULTS_DIR_DATA, f"adam_baseline_{safe}.pt")

        def _create_baseline_if_missing(key, device, task):
            path = _baseline_path_for_key(key)
            lockpath = path + '.lock'
            if os.path.exists(path):
                return path
            # Try to create lock atomically
            try:
                fd = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                have_lock = True
            except FileExistsError:
                have_lock = False
            if not have_lock:
                # Wait for baseline to be created by another worker
                waited = 0
                while not os.path.exists(path) and waited < 600:
                    time.sleep(0.5)
                    waited += 0.5
                return path if os.path.exists(path) else None

            try:
                # set deterministic seed derived from key
                import random as _random
                seed = abs(hash(key)) % (2**32)
                _random.seed(seed)
                np.random.seed(seed)
                try:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
                # enforce deterministic cuDNN for baseline creation
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
                # Build and train Adam-only
                model = PINN(*build_train_data(), param={'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}, layers=task['layers'], device=device, prob_ind=task['p_run'])
                try:
                    model.sweep_param_name = task.get('param_name')
                    model.sweep_param_value = task.get('value')
                except Exception:
                    pass
                for g in model.optimizer_Adam.param_groups:
                    g['lr'] = task['adam_lr']
                # Use ADAM_LOG_EVERY to report Adam-phase progress less frequently
                train_with_cycles(model, task['epochs_adams'], 0, (ADAM_LOG_EVERY if task['epochs_adams'] > 0 else 1), False, cycles=task.get('cycles', 1))
                # save state
                torch.save(model.state_dict(), path)
                return path
            finally:
                try:
                    os.remove(lockpath)
                except Exception:
                    pass

        # Decide whether to reuse Adam baseline
        if task.get('param_name') == 'epochs_lbfgs':
            key = _baseline_key_for_task(task)
            baseline_path = _create_baseline_if_missing(key, device, task)
            if baseline_path is None or not os.path.exists(baseline_path):
                # fallback to full training
                model, dt = train_single_problem(task['p_run'], device, task['epochs_adams'], task['epochs_lbfgs'], task['adam_lr'], task['layers'], relerr_callback=relerr_cb, callback_every=10, sweep_param_name=task.get('param_name'), sweep_param_value=task.get('value'), cycles=task.get('cycles', 1))
                # save trained model state for this exact task
                try:
                    model_path = _model_path_for_key(f"{task['param_name']}|{task['value']}|p{task['p_run']}")
                    torch.save(model.state_dict(), model_path)
                except Exception:
                    pass
            else:
                # create fresh model, load baseline weights, then run LBFGS only
                model = PINN(*build_train_data(), param={'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}, layers=task['layers'], device=device, prob_ind=task['p_run'])
                try:
                    model.sweep_param_name = task.get('param_name')
                    model.sweep_param_value = task.get('value')
                except Exception:
                    pass
                sd = torch.load(baseline_path, map_location=device, weights_only=True)
                model.load_state_dict(sd)
                # run LBFGS only
                start = time.perf_counter()
                train_with_cycles(model, 0, task['epochs_lbfgs'], max(1, math.ceil(task['epochs_lbfgs'] / 10)) if task['epochs_lbfgs'] > 0 else 1, False, callback=relerr_cb, callback_every=10, cycles=task.get('cycles', 1))
                dt = time.perf_counter() - start
                try:
                    model_path = _model_path_for_key(f"{task['param_name']}|{task['value']}|p{task['p_run']}")
                    torch.save(model.state_dict(), model_path)
                except Exception:
                    pass
        else:
            model, dt = train_single_problem(task['p_run'], device, task['epochs_adams'], task['epochs_lbfgs'], task['adam_lr'], task['layers'], relerr_callback=relerr_cb, callback_every=10, sweep_param_name=task.get('param_name'), sweep_param_value=task.get('value'), cycles=task.get('cycles', 1))
            try:
                model_path = _model_path_for_key(f"{task['param_name']}|{task['value']}|p{task['p_run']}")
                torch.save(model.state_dict(), model_path)
            except Exception:
                pass
        preds, exacts, rel_errs = evaluate_model(model, task['p_run'], device)
        final_loss = model.loss_log[-1] if model.loss_log else float('nan')
        meta = {'key': f"{task['param_name']}|{task['value']}|p{task['p_run']}", 'time_sec': dt, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
        save_run_file(meta['key'], {'key': meta['key'], 'meta': meta})
        # save loss CSV with per-component columns so component plots can be produced
        loss_csv = format_filename(task['p_run'], task['param_name'], task['value'], 'loss', dirpath=RESULTS_DIR_DATA, ext='.csv')
        try:
            _write_loss_csv_with_weighted_components(loss_csv, model)
        except Exception:
            pass
        # Also write unified run CSV (epoch, loss components, rel_err mean, tiempo)
        try:
            write_run_csv(task['param_name'], task['value'], task['p_run'], model, dt, data_dir=RESULTS_DIR_DATA)
            run_csv = format_filename(task['p_run'], task['param_name'], task['value'], 'run', dirpath=RESULTS_DIR_DATA, ext='.csv')
            print(f"{MAGENTA}{ICO_SAVE} Guardado CSV: {run_csv}{RESET}")
        except Exception:
            run_csv = None
        # Generate per-run plots: loss vs epochs and rel_err vs epochs (if data present)
        try:
            # Loss plot (single-run overlay)
            plot_path_loss = format_filename(task['p_run'], task['param_name'], task['value'], 'loss', dirpath=RESULTS_DIR_IMGS, ext='.png')
            try:
                plot_utils.plot_loss_overlays([loss_csv], [str(task['value'])], plot_path_loss, smooth_alpha=None)
                print(f"{MAGENTA}{ICO_SAVE} Guardado plot: {plot_path_loss}{RESET}")
                # smoothed per-run loss
                try:
                    plot_path_loss_s = format_filename(task['p_run'], task['param_name'], task['value'], 'loss_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                    plot_utils.plot_loss_overlays([loss_csv], [str(task['value'])], plot_path_loss_s, smooth_alpha=0.12)
                    print(f"{MAGENTA}{ICO_SAVE} Guardado plot (smoothed): {plot_path_loss_s}{RESET}")
                except Exception:
                    pass
            except Exception as e:
                print(f"{YELLOW}Error dibujando loss para {task['param_name']} p{task['p_run']} v{task['value']}: {e}{RESET}")

            # Relerr and R_eff plots: use _read_relerr_csv_and_plot which reads from the relerr CSV
            # (it has R_eff_pred/R_eff_exact columns that run_csv doesn't have)
            relerr_csv = format_filename(task['p_run'], task['param_name'], task['value'], 'relerr', dirpath=RESULTS_DIR_DATA, ext='.csv')
            _read_relerr_csv_and_plot(relerr_csv, task['p_run'], task['param_name'], task['value'], RESULTS_DIR_IMGS, cycles=task.get('cycles', 1))
                    
            # Also try to produce loss components plot from the loss CSV we just wrote
            try:
                plot_path_components = format_filename(task['p_run'], task['param_name'], task['value'], 'loss_components', dirpath=RESULTS_DIR_IMGS, ext='.png')
                comp_csv = _components_csv_for(loss_csv)
                if comp_csv:
                    plot_utils.plot_loss_components(comp_csv, plot_path_components, smooth_alpha=None)
                    print(f"{MAGENTA}{ICO_SAVE} Guardado plot: {plot_path_components}{RESET}")
                # smoothed components
                try:
                    plot_path_components_s = format_filename(task['p_run'], task['param_name'], task['value'], 'loss_components_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                    comp_csv = _components_csv_for(loss_csv)
                    if comp_csv:
                        plot_utils.plot_loss_components(comp_csv, plot_path_components_s, smooth_alpha=0.12)
                        print(f"{MAGENTA}{ICO_SAVE} Guardado plot (smoothed): {plot_path_components_s}{RESET}")
                except Exception:
                    pass
            except Exception as e:
                print(f"{YELLOW}No se pudo dibujar componentes de pérdida para {task['param_name']} p{task['p_run']} v{task['value']}: {e}{RESET}")
        except Exception:
            pass

        return {'param_name': task['param_name'], 'value': task['value'], 'p_run': task['p_run'], 'time_sec': dt, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
    except Exception as e:
        print(f"{RED}{ICO_WARN} Worker failure task={task}: {e}{RESET}")
        return None


def sweep(param_name: str, values: List[Any], fixed: Dict[str, Any], targets: List[int], device: torch.device, layers: List[int], runs_done: Dict[str, Any], overwrite: bool, parallel: bool = False, workers: int = 1, allow_gpu_parallel: bool = False, incremental_epochs: bool = False, cycles: int = 1) -> List[SweepResult]:
    # backward-compat: accept force_retrain via closure if present in fixed
    force_retrain = bool(fixed.get('force_retrain', False))
    try:
        cycles = int(cycles)
    except Exception:
        cycles = 1
    if cycles < 1:
        cycles = 1
    print(f"\n{BOLD}{CYAN}{ICO_GRID} Barrido: {param_name}{RESET} valores={values}")
    results: List[SweepResult] = []
    tasks_to_run: List[Dict[str, Any]] = []
    # Special incremental path for epochs_adams: train cumulatively per target so each larger epoch value reuses previous weights.
    if param_name == 'epochs_adams' and incremental_epochs and not parallel:
        if cycles != 1:
            print(f"{YELLOW}{ICO_WARN} --cycles={cycles} no es compatible con --incremental-epochs para epochs_adams; usando cycles=1 en este barrido.{RESET}")
            cycles = 1
        # Ensure ascending order for cumulative training
        values_sorted = sorted(values, key=lambda x: float(x))
        for p_run in targets:
            # Build / initialize model once
            base_epochs_done = 0
            # Deterministic seed per target for reproducibility
            seed = abs(hash(f"inc_adam|p{p_run}|{layers}|{fixed['adam_lr']}")) % (2**32)
            try:
                import random as _rnd
                _rnd.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            # Build data & model
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2 = build_train_data()
            PARAM = {'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}
            model = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2, param=PARAM, layers=layers, device=device, prob_ind=p_run)
            try:
                model.sweep_param_name = param_name
                model.sweep_param_value = target_epochs
            except Exception:
                pass
            for g in model.optimizer_Adam.param_groups:
                g['lr'] = fixed['adam_lr']
            for target_epochs in values_sorted:
                target_epochs = int(target_epochs)
                if target_epochs < base_epochs_done:
                    continue
                add_epochs = target_epochs - base_epochs_done
                key = f"{param_name}|{target_epochs}|p{p_run}"
                if key in runs_done and not overwrite and not force_retrain:
                    print(f"{GRAY}{ICO_SKIP} Omitiendo {key} (ya completado incremental){RESET}")
                    entry = runs_done.get(key, {})
                    try:
                        meta = entry if 'rel_errs' in entry else entry.get('meta', entry)
                        results.append(SweepResult(param_name, target_epochs, p_run, float(meta.get('time_sec', float('nan'))), float(meta.get('final_loss', float('nan'))), meta.get('rel_errs', [float('nan')]*6), meta.get('preds', [float('nan')]*6)))
                    except Exception:
                        pass
                    base_epochs_done = target_epochs
                    continue
                # If a saved model exists for this milestone, load it and skip retraining unless forced
                model_path = _model_path_for_key(key)
                if os.path.exists(model_path) and not force_retrain:
                    try:
                        model_loaded = PINN(*build_train_data(), param={'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}, layers=layers, device=device, prob_ind=p_run)
                        try:
                            model_loaded.sweep_param_name = param_name
                            model_loaded.sweep_param_value = target_epochs
                        except Exception:
                            pass
                        sd = torch.load(model_path, map_location=device, weights_only=True)
                        model_loaded.load_state_dict(sd)
                        preds, exacts, rel_errs = evaluate_model(model_loaded, p_run, device)
                        final_loss = model_loaded.loss_log[-1] if getattr(model_loaded, 'loss_log', None) else float('nan')
                        meta = {'key': key, 'time_sec': 0.0, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
                        save_run_file(meta['key'], {'key': meta['key'], 'meta': meta})
                        safe_val = str(target_epochs).replace('.', 'p').replace('e', 'e').replace('/', '_')
                        try:
                            write_run_csv(param_name, target_epochs, p_run, model_loaded, 0.0, data_dir=RESULTS_DIR_DATA)
                        except Exception:
                            pass
                        # Also write a per-epoch loss CSV with components for plotting/regeneration
                        try:
                            loss_csv = format_filename(p_run, param_name, target_epochs, 'loss', dirpath=RESULTS_DIR_DATA, ext='.csv')
                            _write_loss_csv_with_weighted_components(loss_csv, model_loaded)
                        except Exception:
                            pass
                        results.append(SweepResult(param_name, target_epochs, p_run, 0.0, final_loss, rel_errs, preds))
                        elapsed_total = time.perf_counter() - (GLOBAL_START or time.perf_counter())
                        print(f"{GREEN}{ICO_OK} [p{p_run}] Skipped (loaded) {param_name}={target_epochs} trained_on=p{p_run} (total {format_elapsed(elapsed_total)}){RESET}")
                        base_epochs_done = target_epochs
                        continue
                    except Exception:
                        pass

                start = time.perf_counter()
                # Prepare relerr callback for this milestone
                # Use force_new when starting fresh (base_epochs_done==0) or force_retrain to ensure correct CSV columns
                try:
                    relerr_cb = _make_relerr_callback(param_name, target_epochs, p_run, device, data_dir=RESULTS_DIR_DATA, every=10, force_new=(base_epochs_done == 0 or force_retrain))
                except Exception:
                    relerr_cb = None
                if add_epochs > 0:
                    # Use ADAM_LOG_EVERY to reduce frequency of Adam logging updates
                    train_with_cycles(model, add_epochs, 0, (ADAM_LOG_EVERY if add_epochs > 0 else 1), False, callback=relerr_cb, callback_every=10, cycles=1)
                    base_epochs_done = target_epochs
                # Optionally run LBFGS fixed phase AFTER each milestone
                if fixed['epochs_lbfgs'] > 0:
                    train_with_cycles(model, 0, fixed['epochs_lbfgs'], max(1, math.ceil(fixed['epochs_lbfgs'] / 10)), False, callback=relerr_cb, callback_every=10, cycles=1)
                dt = time.perf_counter() - start
                preds, exacts, rel_errs = evaluate_model(model, p_run, device)
                final_loss = model.loss_log[-1] if model.loss_log else float('nan')
                meta = {'key': key, 'time_sec': dt, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
                save_run_file(meta['key'], {'key': meta['key'], 'meta': meta})
                safe_val = str(target_epochs).replace('.', 'p').replace('e', 'e').replace('/', '_')
                try:
                    write_run_csv(param_name, target_epochs, p_run, model, dt, data_dir=RESULTS_DIR_DATA)
                except Exception:
                    pass
                # save model state
                try:
                    torch.save(model.state_dict(), model_path)
                except Exception:
                    pass
                results.append(SweepResult(param_name, target_epochs, p_run, dt, final_loss, rel_errs, preds))
                elapsed_total = time.perf_counter() - (GLOBAL_START or time.perf_counter())
                print(f"{GREEN}{ICO_OK} [p{p_run}] Completed {param_name}={target_epochs} trained_on=p{p_run} in {dt:.2f}s (total {format_elapsed(elapsed_total)}){RESET}")
        # After incremental path we proceed to CSV/plots generation below
    else:
        # Build tasks list; incorporate skip logic (original behavior)
        pass
    # Iterate targets outermost so we complete all values for p1, then p2, etc.
    for p_run in targets:
        for i, v in enumerate(values, start=1):
            # defer printing per-task until execution so output aligns with runtime
            # (we still show skips below)
            epochs_adams = fixed['epochs_adams'] if param_name != 'epochs_adams' else int(v)
            epochs_lbfgs = fixed['epochs_lbfgs'] if param_name != 'epochs_lbfgs' else int(v)
            adam_lr = fixed['adam_lr'] if param_name != 'adam_lr' else float(v)
            layers_local = layers if param_name != 'layers' else list(map(int, v.split('-'))) if isinstance(v, str) else v
            key = f"{param_name}|{v}|p{p_run}"
            if key in runs_done and not overwrite and not force_retrain:
                print(f"{GRAY}{ICO_SKIP} Omitiendo {key} (ya completado){RESET}")
                entry = runs_done.get(key, {})
                try:
                    meta = entry if 'rel_errs' in entry else entry.get('meta', entry)
                    results.append(SweepResult(param_name, v, p_run, float(meta.get('time_sec', float('nan'))), float(meta.get('final_loss', float('nan'))), meta.get('rel_errs', [float('nan')]*6), meta.get('preds', [float('nan')]*6)))
                except Exception:
                    pass
                # Ensure per-run CSVs and plots exist by regenerating them from available artifacts.
                try:
                    # Paths
                    run_csv = format_filename(p_run, param_name, v, 'run', dirpath=RESULTS_DIR_DATA, ext='.csv')
                    loss_csv = format_filename(p_run, param_name, v, 'loss', dirpath=RESULTS_DIR_DATA, ext='.csv')
                    rel_csv = format_filename(p_run, param_name, v, 'relerr', dirpath=RESULTS_DIR_DATA, ext='.csv')
                    # If loss CSV missing but model exists, attempt to load model and write loss CSV
                    if (not os.path.exists(loss_csv)):
                        model_path = _model_path_for_key(f"{param_name}|{v}|p{p_run}")
                        if model_path and os.path.exists(model_path):
                            try:
                                model_tmp = PINN(*build_train_data(), param={'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}, layers=layers, device=device, prob_ind=p_run)
                                sd = torch.load(model_path, map_location=device, weights_only=True)
                                model_tmp.load_state_dict(sd)
                                # write loss CSV with components
                                try:
                                    _write_loss_csv_with_weighted_components(loss_csv, model_tmp)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                    # Generate plots from available CSVs
                    try:
                        if os.path.exists(loss_csv):
                            plot_path_loss = format_filename(p_run, param_name, v, 'loss', dirpath=RESULTS_DIR_IMGS, ext='.png')
                            try:
                                plot_utils.plot_loss_overlays([loss_csv], [str(v)], plot_path_loss, smooth_alpha=None)
                            except Exception:
                                pass
                            # components
                            try:
                                    plot_path_components = format_filename(p_run, param_name, v, 'loss_components', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                    comp_csv = _components_csv_for(loss_csv)
                                    if comp_csv:
                                        plot_utils.plot_loss_components(comp_csv, plot_path_components, smooth_alpha=None)
                            except Exception:
                                pass
                            # smoothed variants
                            try:
                                plot_path_loss_s = format_filename(p_run, param_name, v, 'loss_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                plot_utils.plot_loss_overlays([loss_csv], [str(v)], plot_path_loss_s, smooth_alpha=0.12)
                            except Exception:
                                pass
                            try:
                                plot_path_components_s = format_filename(p_run, param_name, v, 'loss_components_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                comp_csv = _components_csv_for(loss_csv)
                                if comp_csv:
                                    plot_utils.plot_loss_components(comp_csv, plot_path_components_s, smooth_alpha=0.12)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Regenerate relerr plot from run_csv if present
                    try:
                        if os.path.exists(run_csv):
                            xs = []; ys = []
                            with open(run_csv, 'r', newline='') as fh:
                                r = csv.reader(fh)
                                headers = next(r)
                                try:
                                    idx_epoch = headers.index('epoch')
                                    idx_rel = headers.index('rel_err')
                                except Exception:
                                    idx_epoch = 0; idx_rel = 5
                                for row in r:
                                    if not row:
                                        continue
                                    try:
                                        ep = int(row[idx_epoch])
                                    except Exception:
                                        continue
                                    rel = row[idx_rel].strip()
                                    if rel == '':
                                        continue
                                    try:
                                        relf = float(rel)
                                    except Exception:
                                        continue
                                    xs.append(ep); ys.append(relf)
                            if xs and ys:
                                plot_path_rel = format_filename(p_run, param_name, v, 'relerr', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                try:
                                    plot_utils.plot_relerr(xs, ys, 'Epoch', f"RelErr p{p_run} {param_name}={v}", plot_path_rel, logx=False, smooth_alpha=None)
                                except Exception:
                                    pass
                                # smoothed relerr
                                try:
                                    plot_path_rel_s = format_filename(p_run, param_name, v, 'relerr_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                                    plot_utils.plot_relerr(xs, ys, 'Epoch', f"RelErr p{p_run} {param_name}={v}", plot_path_rel_s, logx=False, smooth_alpha=0.12)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass
                continue
            tasks_to_run.append({'param_name': param_name, 'value': v, 'p_run': p_run, 'epochs_adams': epochs_adams, 'epochs_lbfgs': epochs_lbfgs, 'adam_lr': adam_lr, 'layers': layers_local, 'force_retrain': force_retrain, 'cycles': cycles})

    # Execute tasks
    total_tasks = len(tasks_to_run)
    value_total = len(values)
    targets_count = len(targets)
    if param_name == 'epochs_adams' and incremental_epochs and not parallel:
        pass  # already executed incremental path
    elif parallel and total_tasks > 0:
        if device.type == 'cuda' and workers > 1 and not allow_gpu_parallel:
            print(f"{YELLOW}{ICO_WARN} GPU detected; multiple workers on same GPU can cause contention. Use --allow-gpu-parallel to override. Running sequentially.{RESET}")
            use_workers = 1
        else:
            use_workers = workers
        # print plan
        print(f"{DIM}Ejecutando {total_tasks} tareas en paralelo (workers={use_workers}){RESET}")
        ctx = get_context('spawn')
        with ctx.Pool(processes=use_workers) as pool:
            mapped = pool.map(_worker_run_task, [(t, device.type) for t in tasks_to_run])
        for res in mapped:
            if res is None:
                continue
            results.append(SweepResult(res['param_name'], res['value'], res['p_run'], res['time_sec'], res['final_loss'], res['rel_errs'], res['preds']))
            elapsed_total = time.perf_counter() - (GLOBAL_START or time.perf_counter())
            print(f"{GREEN}{ICO_OK} Completed {res['param_name']}={res['value']} trained_on=p{res['p_run']} in {res['time_sec']:.2f}s (total {format_elapsed(elapsed_total)}){RESET}")
    elif total_tasks > 0:
        # Run sequentially and print progress per executed task
        for idx, task in enumerate(tasks_to_run, start=1):
            # compute which value index this task belongs to (1..value_total)
            v_idx = ((idx - 1) // targets_count) + 1 if targets_count > 0 else idx
            res = _worker_run_task((task, device.type))
            if res is None:
                continue
            results.append(SweepResult(res['param_name'], res['value'], res['p_run'], res['time_sec'], res['final_loss'], res['rel_errs'], res['preds']))
            elapsed_total = time.perf_counter() - (GLOBAL_START or time.perf_counter())
            print(f"{GREEN}{ICO_OK} [{v_idx}/{value_total}] Completed {res['param_name']}={res['value']} trained_on=p{res['p_run']} in {res['time_sec']:.2f}s (total {format_elapsed(elapsed_total)}){RESET}")
    # Save CSV
    # Save combined CSV with rel_errs for all 6 coefficients
    csv_path = os.path.join(RESULTS_DIR_DATA, f"sweep_{param_name}.csv")
    headers = ["param", "value", "trained_target", "time_sec", "final_loss"] + [f"rel_err_p{i+1}_pct" for i in range(6)] + [f"pred_p{i+1}" for i in range(6)]
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            row = [r.param, r.value, r.trained_target, f"{r.time_sec:.4f}", f"{r.final_loss:.6e}"] + [f"{x:.6f}" for x in r.rel_errs] + [f"{x:.6f}" for x in r.preds]
            w.writerow(row)
    print(f"{MAGENTA}{ICO_SAVE} Guardado CSV: {csv_path}{RESET}")

    # Plot if matplotlib available - produce one rel-err curve per trained target, and use log scale
    try:
        import matplotlib.pyplot as plt
        # Aggregate values and group by trained_target
        by_target = {}
        for r in results:
            by_target.setdefault(r.trained_target, []).append(r)
        # For each trained target produce rel_err vs value plot (single curve for that p)
        for t, items in by_target.items():
            xs_raw = [it.value for it in items]
            # try to convert numeric x values for plotting, otherwise use indices
            def _to_numeric_list(lst):
                nums = []
                is_numeric = True
                for x in lst:
                    try:
                        nums.append(float(x))
                    except Exception:
                        is_numeric = False
                        break
                return (nums, is_numeric)

            xs_vals, xs_numeric = _to_numeric_list(xs_raw)
            ys = [it.rel_errs[t-1] for it in items]
            # Determine x values and whether to use log-x. Force log-x for epochs_adams.
            if xs_numeric:
                xs_plot = xs_vals
                if param_name == 'epochs_adams':
                    logx = True
                else:
                    logx = _is_exponential_numeric(xs_vals)
            else:
                xs_plot = list(range(1, len(xs_raw)+1))
                logx = False

            plot_path = format_filename(t, param_name, 'sweep', 'relerr_log', dirpath=RESULTS_DIR_IMGS, ext='.png')
            title = f'RelErr (p{t}) vs {param_name} (trained_on p{t})'
            try:
                plot_utils.plot_relerr(xs_plot, ys, param_name, title, plot_path, logx=logx, smooth_alpha=None)
                print(f"{MAGENTA}{ICO_SAVE} Guardado plot: {plot_path}{RESET}")
            except Exception as e:
                print(f"{YELLOW}Error al dibujar relerr para {param_name} p{t}: {e}{RESET}")
        # Loss overlay: for each trained target, overlay loss curves for different values (log y-scale)
        for t, items in by_target.items():
            loss_paths = []
            labels = []
            for it in items:
                loss_csv = format_filename(t, param_name, it.value, 'run', dirpath=RESULTS_DIR_DATA, ext='.csv')
                if os.path.exists(loss_csv):
                    loss_paths.append(loss_csv)
                    labels.append(str(it.value))
            if not loss_paths:
                continue
            plot_path = format_filename(t, param_name, 'sweep', 'losses_log', dirpath=RESULTS_DIR_IMGS, ext='.png')
            try:
                plot_utils.plot_loss_overlays(loss_paths, labels, plot_path, smooth_alpha=None)
                print(f"{MAGENTA}{ICO_SAVE} Guardado plot: {plot_path}{RESET}")
                smooth_path = format_filename(t, param_name, 'sweep', 'losses_log_smooth', dirpath=RESULTS_DIR_IMGS, ext='.png')
                plot_utils.plot_loss_overlays(loss_paths, labels, smooth_path, smooth_alpha=0.12)
                print(f"{MAGENTA}{ICO_SAVE} Guardado plot (smoothed): {smooth_path}{RESET}")
            except Exception as e:
                print(f"{YELLOW}Error al dibujar losses para {param_name} p{t}: {e}{RESET}")
    except ImportError:
        print(f"{YELLOW}matplotlib/pandas no disponible; omitiendo plots para {param_name}{RESET}")
    return results


def parse_list(arg: str, cast=float) -> List[Any]:
    return [cast(x.strip()) for x in arg.split(',') if x.strip()]


def _is_exponential_numeric(arr: List[float]) -> bool:
    """Return True when numeric list `arr` is positive and its consecutive
    ratios are approximately constant (geometric progression), indicating
    an exponential/scaled sweep (suitable for log-x plotting).
    """
    if len(arr) < 2:
        return False
    if any(x <= 0 for x in arr):
        return False
    ratios = []
    for i in range(len(arr) - 1):
        try:
            ratios.append(arr[i+1] / arr[i])
        except Exception:
            return False
    if not ratios:
        return False
    import statistics as _st
    mean_r = _st.mean(ratios)
    if mean_r <= 0:
        return False
    try:
        std_r = _st.pstdev(ratios)
    except Exception:
        std_r = 0.0
    cov = std_r / mean_r if mean_r != 0 else float('inf')
    return cov < 0.5


def regenerate_plots(results_dir: str):
    """Regenerate plots from existing sweep CSVs and loss CSVs in `results_dir`.
    Produces log-scale relerr and loss plots per parameter and trained target.
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        pd = None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print(f"{YELLOW}matplotlib/pandas no disponible; no se pueden regenerar plots{RESET}")
            return

    # process each sweep CSV
    for fname in os.listdir(results_dir):
        if not fname.startswith('sweep_') or not fname.endswith('.csv'):
            continue
        path = os.path.join(results_dir, fname)
        try:
            if pd is not None:
                df = pd.read_csv(path)
            else:
                # fallback CSV reader
                import csv as _csv
                with open(path, 'r', newline='') as fh:
                    r = _csv.reader(fh)
                    headers = next(r)
                    rows = list(r)
                # build simple dict
                df = None
        except Exception as e:
            print(f"{RED}Error leyendo {path}: {e}{RESET}")
            continue

        param_name = fname[len('sweep_'):-len('.csv')]
        # figure out trained targets in this CSV
        if pd is not None:
            tset = sorted(df['trained_target'].unique())
        else:
            # fallback parse trained_target column index
            try:
                idx = headers.index('trained_target')
                tset = sorted(set(int(r[idx]) for r in rows))
            except Exception:
                tset = []

        for t in tset:
            # filter rows for this trained target
            if pd is not None:
                df_t = df[df['trained_target'] == t]
                xs_raw = df_t['value'].tolist()
                ycol = f'rel_err_p{t}_pct'
                ys = df_t[ycol].astype(float).tolist()
            else:
                # fallback: scan rows
                xs_raw = []
                ys = []
                try:
                    val_idx = headers.index('value')
                    ycol_idx = headers.index(f'rel_err_p{t}_pct')
                    for r in rows:
                        if int(r[headers.index('trained_target')]) == t:
                            xs_raw.append(r[val_idx])
                            ys.append(float(r[ycol_idx]))
                except Exception:
                    continue

            # try numeric x
            xs_num = None
            try:
                xs_num = [float(x) for x in xs_raw]
            except Exception:
                xs_num = None

            def _is_exponential_numeric(arr: List[float]) -> bool:
                if len(arr) < 2:
                    return False
                if any(x <= 0 for x in arr):
                    return False
                # ratios between consecutive elements
                ratios = []
                for i in range(len(arr) - 1):
                    try:
                        ratios.append(arr[i+1] / arr[i])
                    except Exception:
                        return False
                if not ratios:
                    return False
                import statistics as _st
                mean_r = _st.mean(ratios)
                if mean_r <= 0:
                    return False
                try:
                    std_r = _st.pstdev(ratios)
                except Exception:
                    std_r = 0.0
                # if ratios are similar (low coeff of variation), consider exponential
                cov = std_r / mean_r if mean_r != 0 else float('inf')
                return cov < 0.5

            # choose x values and whether to use log-x
            if xs_num is not None:
                xs_plot = xs_num
                if param_name == 'epochs_adams':
                    logx = True
                else:
                    logx = _is_exponential_numeric(xs_num)
            else:
                xs_plot = list(range(1, len(xs_raw)+1))
                logx = False
            outp = format_filename(t, param_name, 'sweep', 'relerr_log', dirpath=results_dir, ext='.png')
            title = f'RelErr (p{t}) vs {param_name} (trained_on p{t})'
            try:
                plot_utils.plot_relerr(xs_plot, ys, param_name, title, outp, logx=logx, smooth_alpha=None)
                print(f"{MAGENTA}{ICO_SAVE} Regenerado plot: {outp}{RESET}")
            except Exception as e:
                print(f"{YELLOW}Error regenerando relerr plot {param_name} p{t}: {e}{RESET}")

            # losses overlay using loss CSV files
            loss_paths = []
            labels = []
            for i, val in enumerate(xs_raw):
                loss_csv = os.path.join(results_dir, f"p{t}_{param_name}_{str(val).replace('.', 'p').replace('/', '_')}_loss.csv")
                if os.path.exists(loss_csv):
                    loss_paths.append(loss_csv)
                    labels.append(str(val))
            if not loss_paths:
                continue
            outp2 = format_filename(t, param_name, 'sweep', 'losses_log', dirpath=results_dir, ext='.png')
            try:
                plot_utils.plot_loss_overlays(loss_paths, labels, outp2, smooth_alpha=None)
                print(f"{MAGENTA}{ICO_SAVE} Regenerado plot: {outp2}{RESET}")
                outp2s = format_filename(t, param_name, 'sweep', 'losses_log_smooth', dirpath=results_dir, ext='.png')
                plot_utils.plot_loss_overlays(loss_paths, labels, outp2s, smooth_alpha=0.12)
                print(f"{MAGENTA}{ICO_SAVE} Regenerado plot (smoothed): {outp2s}{RESET}")
            except Exception as e:
                print(f"{YELLOW}Error regenerando losses plot {param_name} p{t}: {e}{RESET}")


def _refine_with_default_lbfgs(p_target: int, device: torch.device, layers: List[int], default_lbfgs: int, cycles: int = 1) -> Dict[str, Any]:
    """Refine the latest Adam-only milestone for p_target with default LBFGS epochs.
    Tries to load the model saved at the largest epochs_adams milestone and run LBFGS only.
    Saves a new model under key "epochs_lbfgs|<default>|p<p_target>" and returns metadata.
    """
    try:
        # Find largest Adam milestone model for this p
        adam_models = []
        for fname in os.listdir(RESULTS_DIR_DATA):
            if not fname.startswith('model_') or not fname.endswith('.pt'):
                continue
            # filename encodes key inside: model_<safe>.pt where safe corresponds to f"{param}|{val}|p<p>"
            # we'll match those starting with epochs_adams
            safe = fname[len('model_'):-len('.pt')]
            safe_key = safe.replace('_', '|')  # approximate reverse; we also replaced ':' and '/' earlier
            # Instead of reverse mapping, parse original per-run JSONs
            pass
    except Exception:
        pass
    # Robust approach: scan per-run JSON files for epochs_adams entries for this p and pick max value
    max_epoch = None
    # Scan per-run JSON files in data dir (new format ends with '_run.json')
    for fname in os.listdir(RESULTS_DIR_DATA):
        if not (fname.endswith('_run.json') or (fname.startswith('run_') and fname.endswith('.json'))):
            continue
        try:
            with open(os.path.join(RESULTS_DIR_DATA, fname), 'r', encoding='utf-8') as fh:
                entry = json.load(fh)
            meta = entry if 'rel_errs' in entry else entry.get('meta', entry)
            key = meta.get('key', '')
            if not key.startswith('epochs_adams|'):
                continue
            # key format: epochs_adams|<val>|p<p>
            parts = key.split('|')
            if len(parts) < 3:
                continue
            if not parts[2].startswith('p'):
                continue
            pnum = int(parts[2][1:])
            if pnum != p_target:
                continue
            try:
                val = int(parts[1])
            except Exception:
                continue
            max_epoch = val if (max_epoch is None or val > max_epoch) else max_epoch
        except Exception:
            continue
    if max_epoch is None:
        print(f"{YELLOW}{ICO_WARN} No se encontró checkpoint Adam para p={p_target}; omitiendo refinamiento LBFGS{RESET}")
        return {}
    # Load model from the saved state corresponding to this milestone
    adam_key = f"epochs_adams|{max_epoch}|p{p_target}"
    adam_model_path = _model_path_for_key(adam_key)
    if not os.path.exists(adam_model_path):
        print(f"{YELLOW}{ICO_WARN} Falta modelo Adam {adam_model_path}; omitiendo refinamiento LBFGS{RESET}")
        return {}
    # Build model and load state
    y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2 = build_train_data()
    PARAM = {'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}
    model = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
                 param=PARAM, layers=layers, device=device, prob_ind=p_target)
    try:
        model.sweep_param_name = 'epochs_adams'
        model.sweep_param_value = max_epoch
    except Exception:
        pass
    try:
        sd = torch.load(adam_model_path, map_location=device, weights_only=True)
    except TypeError:
        # compatibility: older torch versions may not support weights_only
        sd = torch.load(adam_model_path, map_location=device)
    model.load_state_dict(sd)
    start = time.perf_counter()
    try:
        relerr_cb = _make_relerr_callback('epochs_lbfgs', default_lbfgs, p_target, device, data_dir=RESULTS_DIR_DATA, every=10, force_new=True)
    except Exception:
        relerr_cb = None
    # cycles here repeats the LBFGS refinement (useful if it stalls early)
    train_with_cycles(model, 0, default_lbfgs, max(1, math.ceil(default_lbfgs / 10)) if default_lbfgs > 0 else 1, False,
                      callback=relerr_cb, callback_every=10, cycles=cycles)
    dt = time.perf_counter() - start
    # Evaluate and save
    preds, exacts, rel_errs = evaluate_model(model, p_target, device)
    final_loss = model.loss_log[-1] if getattr(model, 'loss_log', None) else float('nan')
    lbfgs_key = f"epochs_lbfgs|{default_lbfgs}|p{p_target}"
    try:
        torch.save(model.state_dict(), _model_path_for_key(lbfgs_key))
    except Exception:
        pass
    meta = {'key': lbfgs_key, 'time_sec': dt, 'final_loss': final_loss, 'rel_errs': rel_errs, 'preds': preds}
    save_run_file(lbfgs_key, {'key': lbfgs_key, 'meta': meta})
    try:
        write_run_csv('epochs_lbfgs', default_lbfgs, p_target, model, dt, data_dir=RESULTS_DIR_DATA)
    except Exception:
        pass
    return meta


def main():
    parser = argparse.ArgumentParser(description="Grid (sequential) hyperparameter sweeps for PINN")
    parser.add_argument('--targets', type=str, default=','.join(map(str, SWEEP_CONFIG.get('targets', list(range(1,7))))), help='Comma-separated target problems to evaluate (default from config.SWEEP_CONFIG)')
    parser.add_argument('--target', type=int, default=None, choices=range(1,7), help='(deprecated) single problem target; if provided, overrides --targets')
    parser.add_argument('--epochs-adams', type=str, default=','.join(map(str, SWEEP_CONFIG.get('epochs_adams', [EPOCH_ADAMS]))), help='Lista de valores para Adam epochs (default from config)')
    parser.add_argument('--epochs-lbfgs', type=str, default=','.join(map(str, SWEEP_CONFIG.get('epochs_lbfgs', [EPOCH_LBFGS]))), help='Lista de valores para LBFGS epochs (default from config)')
    parser.add_argument('--adam-lr', type=str, default=','.join(map(str, SWEEP_CONFIG.get('adam_lr', [ADAM_LR]))), help='Lista de valores para Adam LR (default from config)')
    parser.add_argument('--layers', type=str, default=','.join(SWEEP_CONFIG.get('layers', ['-'.join(map(str, LAYERS))])), help='Lista de arquitecturas separadas por coma, cada una como "1-50-50-3" (default from config)')
    parser.add_argument('--n-collocation', type=str, default=','.join(map(str, SWEEP_CONFIG.get('n_collocation', [N_COLLOCATION]))), help='Valores para N_COLLOCATION (se usa reconstruyendo datos)')
    parser.add_argument('--no-lbfgs', action='store_true', help='Skip LBFGS phase (set LBFGS epochs to 0) to speed up quick sweeps')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run state (reads grid_results/runs_done.json and skips completed runs)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing completed run entries and rerun them')
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel worker processes')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers to use (default 2)')
    parser.add_argument('--allow-gpu-parallel', action='store_true', help='Allow multiple workers using the same GPU (may slow/contend)')
    parser.add_argument('--regen-plots', action='store_true', help='Only (re)generate plots from existing CSV/JSON without retraining')
    parser.add_argument('--quick', action='store_true', help='Modo rápido: recorta listas a los primeros 2 valores')
    parser.add_argument('--full-sequence', action='store_true', help='Entrenar secuencia completa en cada configuración (muy costoso)')
    parser.add_argument('--incremental-epochs', dest='incremental_epochs', action='store_true', help='Entrenar epochs_adams de forma incremental reutilizando pesos')
    parser.add_argument('--no-incremental-epochs', dest='incremental_epochs', action='store_false', help='No reutilizar pesos en barrido de epochs_adams (comportamiento por defecto antiguo)')
    parser.add_argument('--force-retrain', action='store_true', help='Force retrain even if a saved model exists for a task')
    parser.add_argument('--cycles', type=int, default=1, help='Número de ciclos de entrenamiento (repite Adam+LBFGS) por ejecución (default: 1)')
    # ==========================================================================
    # SOLUCIONES PARA MEJORAR CORRELACIÓN LOSS-ERROR RELATIVO
    # ==========================================================================
    parser.add_argument('--gradient-reg', action='store_true', 
                        help='[Solucion 3] Habilitar regularizacion del gradiente (penaliza curvatura excesiva de chi)')
    parser.add_argument('--gradient-reg-weight', type=float, default=1e-4,
                        help='[Solucion 3] Peso del termino de regularizacion del gradiente (default: 1e-4)')
    parser.add_argument('--fourier-features', action='store_true',
                        help='[Solucion 5] Habilitar Fourier Feature Embedding para capturar multiples escalas')
    parser.add_argument('--fourier-scales', type=str, default='1.0,2.0,4.0,8.0,16.0',
                        help='[Solucion 5] Escalas de frecuencia para Fourier features (default: 1.0,2.0,4.0,8.0,16.0)')
    parser.add_argument('--uniqueness-weight-min', type=float, default=1.0,
                        help='[Solucion 4] Peso minimo para condicion de unicidad (default: 1.0, mayor fuerza integral=0)')
    parser.add_argument('--voigt-reuss', action='store_true',
                        help='[Solucion 6A] Habilitar penalizacion por cotas de Voigt-Reuss (proxy fisico no tramposo)')
    parser.add_argument('--voigt-reuss-weight', type=float, default=1.0,
                        help='[Solucion 6A] Peso del termino de penalizacion Voigt-Reuss (default: 1.0)')
    parser.add_argument('--voigt-reuss-margin', type=float, default=0.0,
                        help='[Solucion 6A] Margen para estrechar las cotas (0.0=cotas exactas, 0.1=10%% mas estricto)')
    parser.add_argument('--integral-consistency', action='store_true',
                        help='[Solucion 6B] Habilitar penalizacion por consistencia de integral (proxy no tramposo)')
    parser.add_argument('--integral-consistency-weight', type=float, default=1.0,
                        help='[Solucion 6B] Peso del termino de consistencia de integral (default: 1.0)')
    parser.set_defaults(incremental_epochs=True)
    args = parser.parse_args()
    
    # Aplicar configuraciones de las soluciones 3, 4, 5 al módulo de configuración
    import pinn.config as cfg
    if args.gradient_reg:
        cfg.GRADIENT_REGULARIZATION = True
        cfg.GRADIENT_REG_WEIGHT = args.gradient_reg_weight
        print(f"{CYAN}[Solución 3] Regularización del gradiente ACTIVADA (peso={args.gradient_reg_weight}){RESET}")
    if args.fourier_features:
        cfg.FOURIER_FEATURES = True
        cfg.FOURIER_SCALES = [float(x) for x in args.fourier_scales.split(',')]
        print(f"{CYAN}[Solución 5] Fourier Features ACTIVADO (escalas={cfg.FOURIER_SCALES}){RESET}")
    cfg.WEIGHT_UNIQUENESS_MIN = args.uniqueness_weight_min
    if args.uniqueness_weight_min > 0.1:
        print(f"{CYAN}[Solución 4] Peso mínimo de Uniqueness aumentado a {args.uniqueness_weight_min}{RESET}")
    if args.voigt_reuss:
        cfg.VOIGT_REUSS_PENALTY = True
        cfg.VOIGT_REUSS_WEIGHT = args.voigt_reuss_weight
        cfg.VOIGT_REUSS_MARGIN = args.voigt_reuss_margin
        print(f"{CYAN}[Solución 6A] Penalización Voigt-Reuss ACTIVADA (peso={args.voigt_reuss_weight}, margen={args.voigt_reuss_margin}){RESET}")
    if args.integral_consistency:
        cfg.INTEGRAL_CONSISTENCY_PENALTY = True
        cfg.INTEGRAL_CONSISTENCY_WEIGHT = args.integral_consistency_weight
        print(f"{CYAN}[Solución 6B] Consistencia de Integral ACTIVADA (peso={args.integral_consistency_weight}){RESET}")

    device = get_device()
    print(f"{BOLD}{CYAN}{ICO_RUN} Dispositivo: {device}{RESET} DTYPE={DTYPE}")
    # record global start time for total elapsed reporting
    global GLOBAL_START
    GLOBAL_START = time.perf_counter()

    # Base fixed values
    fixed = {
        'epochs_adams': EPOCH_ADAMS,
        'epochs_lbfgs': EPOCH_LBFGS,
        'adam_lr': ADAM_LR,
        'layers': LAYERS,
    }

    if args.no_lbfgs:
        fixed['epochs_lbfgs'] = 0
    runs_done = load_runs_done() if args.resume else {}
    # Setup parallel options with smarter defaults:
    # - If user explicitly passed flags, respect them.
    # - Default: enable parallel on CPU/MPS (workers = cpu_count-1), disable on CUDA (use single worker to avoid contention on RTX4090).
    user_provided_parallel = any(x == '--parallel' for x in os.sys.argv)
    user_provided_workers = any(x.startswith('--workers') for x in os.sys.argv)
    allow_gpu_parallel = args.allow_gpu_parallel
    if user_provided_parallel:
        parallel = args.parallel
    else:
        parallel = (device.type != 'cuda')
    if user_provided_workers:
        workers = max(1, args.workers)
    else:
        workers = 1 if device.type == 'cuda' else max(1, (os.cpu_count() or 2) - 1)
    print(f"{DIM}Parallel default -> parallel={parallel}, workers={workers}, allow_gpu_parallel={allow_gpu_parallel}{RESET}")

    # Determine targets list (allow deprecated --target)
    if args.target is not None:
        targets = [args.target]
    else:
        targets = [int(x) for x in args.targets.split(',') if x.strip()]

    # Parse sweeps
    sweeps: Dict[str, List[Any]] = {
        'epochs_adams': parse_list(args.epochs_adams, int),
        'epochs_lbfgs': parse_list(args.epochs_lbfgs, int),
        'adam_lr': parse_list(args.adam_lr, float),
        'n_collocation': parse_list(args.n_collocation, int),
    }
    if args.layers:
        sweeps['layers'] = [s.strip() for s in args.layers.split(',') if s.strip()]

    if args.quick:
        for k in sweeps:
            sweeps[k] = sweeps[k][:2]
        print(f"{YELLOW}Modo rápido activado: usando sólo primeros 2 valores de cada lista{RESET}")

    if args.regen_plots:
        print(f"{BOLD}{CYAN}Regenerando plots desde CSV/JSON existentes en {RESULTS_DIR}{RESET}")
        regenerate_plots(RESULTS_DIR)
        return

    # Asegurar que runs_done incluya los JSON por-ejecución ya guardados
    try:
        merge_run_files()
    except Exception:
        pass
    runs_all = load_runs_done()
    # Usar siempre el estado actual como base de "runs_done" para skips
    runs_done = runs_all

    # Ejecutar por problema: primero p1, luego p2, etc.
    targets = sorted(targets)
    all_results: Dict[str, List[SweepResult]] = {}
    incremental_flag = args.incremental_epochs

    def _has_all_runs_for_p(pnum: int) -> bool:
        for pname, vals in sweeps.items():
            for v in vals:
                if f"{pname}|{v}|p{pnum}" not in runs_all:
                    return False
        return True

    for p in targets:
        print(f"\n{BOLD}{CYAN}{ICO_RUN} Analizando problema p={p}{RESET}")
        if (not args.force_retrain) and _has_all_runs_for_p(p) and not args.overwrite:
            print(f"{GRAY}{ICO_SKIP} p={p}: todos los barridos encontrados. Rehaciendo sólo gráficas.{RESET}")
            regenerate_plots(RESULTS_DIR)
            continue

        # 1) Barrido de epochs_adams (incremental, sólo Adam) — ejecutar sólo
        # si hay más de un valor pedido. Si sólo hay un valor (por defecto),
        # omitimos este paso y dejamos que los barridos específicos como
        # `n_collocation` manejen el entrenamiento completo por valor.
        adams_values = sweeps.get('epochs_adams', [])
        # Inform when skipping the special epochs_adams sweep (only run when multiple values requested)
        if len(adams_values) > 1:
            fixed_adam_only = {**fixed, 'epochs_lbfgs': 0}
            res_list = sweep('epochs_adams', adams_values, {**fixed_adam_only, 'force_retrain': args.force_retrain}, [p], device, fixed['layers'], runs_done, args.overwrite, parallel=False, workers=1, allow_gpu_parallel=allow_gpu_parallel, incremental_epochs=True, cycles=args.cycles)
            all_results.setdefault('epochs_adams', []).extend(res_list)
            # Refinamiento LBFGS por defecto una vez completado el mayor Adam
            meta = _refine_with_default_lbfgs(p, device, fixed['layers'], fixed['epochs_lbfgs'], cycles=args.cycles)
            if meta:
                elapsed_total = time.perf_counter() - (GLOBAL_START or time.perf_counter())
                print(f"{GREEN}{ICO_OK} [p{p}] Refinado LBFGS por defecto (epochs_lbfgs={fixed['epochs_lbfgs']}) en {format_elapsed(meta.get('time_sec', 0.0))} (total {format_elapsed(elapsed_total)}){RESET}")
        else:
            print(f"{DIM}{ICO_WARN} Saltando barrido de 'epochs_adams' para p={p} (solo 1 valor configurado){RESET}")

        # 2) Barridos del resto de parámetros para p (usa implementación general)
        for pname in ['epochs_lbfgs', 'adam_lr', 'n_collocation', 'layers']:
            vals = sweeps.get(pname, [])
            if not vals:
                # Explicitly inform when a sweep parameter is not configured
                print(f"{DIM}{ICO_WARN} Saltando barrido de '{pname}' para p={p} (no configurado){RESET}")
                continue
            if len(vals) <= 1:
                print(f"{DIM}{ICO_WARN} Saltando barrido de '{pname}' para p={p} (solo 1 valor configurado){RESET}")
                continue
            res_list = sweep(pname, vals, {**fixed, 'force_retrain': args.force_retrain}, [p], device, fixed['layers'], runs_done, args.overwrite, parallel=parallel, workers=workers, allow_gpu_parallel=allow_gpu_parallel, incremental_epochs=False, cycles=args.cycles)
            all_results.setdefault(pname, []).extend(res_list)

    print(f"\n{BOLD}{GREEN}{ICO_OK} Finalizado todos los barridos{RESET}")
    print(f"Resultados en carpeta: {RESULTS_DIR}")

if __name__ == '__main__':
    main()
