#!/usr/bin/env python3
"""
PINN Final - Entrenamiento simplificado de PINNs para compuestos laminados.

Este script realiza el entrenamiento usando los parámetros definidos en pinn/config.py
sin búsqueda de hiperparámetros. Mantiene la estructura de ciclos (Adam + LBFGS)
con un criterio de parada adaptativo basado en la convergencia de R_eff.

Criterio de parada:
  - A partir del ciclo 5, se calcula la media de los R_eff predichos al final de cada ciclo
  - A partir del ciclo 10, si |media_actual - media_hace_5_ciclos| < tolerancia, se para
  - El valor final de R_eff se determina según:
    * Si los últimos 5 ciclos son monótonos: se usa el último valor
    * Si los últimos 5 ciclos oscilan: se usa la media de los últimos 5

Modo Multi-Run (--n-runs > 1):
  - Ejecuta N iteraciones completas con diferentes seeds
  - Evita convergencia a óptimos locales por mala suerte
  - Devuelve estadísticas robustas: media, media truncada y mediana
  - Incluye todos los R_eff diagonales y cruzados
  - Guarda todos los Reff para cálculos posteriores (CSV y JSON)

Uso:
  python pinn_final.py                      # Entrenar todos los problemas (1-6)
  python pinn_final.py --target 3           # Entrenar solo p=3
  python pinn_final.py --targets 1,2,3      # Entrenar p=1, p=2, p=3
  python pinn_final.py --tolerance 0.01     # Tolerancia de convergencia 0.01%
  python pinn_final.py --force-retrain      # Forzar reentrenamiento aunque existan modelos
  python pinn_final.py --n-runs 10          # Ejecutar 10 iteraciones con diferentes seeds
  python pinn_final.py --n-runs 10 --base-seed 42  # Multi-run con seed base específico
  python pinn_final.py --quiet-iterations   # Silenciar mensajes de iteración Adam/LBFGS
  python pinn_final.py --truncate-pct 25    # Media truncada: elimina 25% por cada lado
"""

import os

import time
import csv
import argparse
import math
import random
import statistics
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

from pinn.device import get_device
from pinn.config import (
    LAYERS, EPOCH_ADAMS, EPOCH_LBFGS, ADAM_LR, N_COLLOCATION,
    VF_mat1, VF_mat2, DTYPE, TRAIN_SEED,
    EPOCH_ADAMS_FOLLOWING_CYCLES, EPOCH_LBFGS_FOLLOWING_CYCLES,
    ADAM_LATERCYCLES_PRINT_EVERY, LBFGS_LATERCYCLES_PRINT_EVERY,
    ADAM_PRINT_EVERY
)
from pinn import plot_utils
from pinn.materials import (
    R_mat1, R_mat2,
    R11_mat1, R12_mat1, R13_mat1, R22_mat1, R23_mat1, R33_mat1, R44_mat1, R55_mat1, R66_mat1,
    R11_mat2, R12_mat2, R13_mat2, R22_mat2, R23_mat2, R33_mat2, R44_mat2, R55_mat2, R66_mat2,
)
from pinn.data import build_train_data, build_eval_points
from pinn.model import PINN

# ==============================================================================
# ANSI colors/icons para logs
# ==============================================================================
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"
MAGENTA = "\033[35m"; CYAN = "\033[36m"; GRAY = "\033[90m"
ICO_RUN = "🚀"; ICO_OK = "✅"; ICO_SAVE = "💾"; ICO_TRAIN = "🏋️"; ICO_STOP = "🛑"; ICO_CYCLE = "🔄"

# Tolerancia de convergencia por defecto (en porcentaje)
DEFAULT_CONVERGENCE_TOLERANCE = 0.05  # 0.05%
# Número mínimo de ciclos antes de empezar a evaluar convergencia
MIN_CYCLES_BEFORE_CHECK = 10
# Ventana de ciclos para calcular media y evaluar monotonía
CONVERGENCE_WINDOW = 5
# Número de iteraciones por defecto para multi-run
DEFAULT_N_RUNS = 10
# Porcentaje de valores extremos a quitar para media truncada (por cada lado)
# Por ejemplo, 20 = 20%, si N=20 se quitan 4 arriba y 4 abajo
DEFAULT_TRUNCATE_PCT = 20


# ==============================================================================
# Configuración determinista para reproducibilidad
# ==============================================================================
def setup_deterministic(seed: int = TRAIN_SEED) -> torch.device:
    """
    Configura el entorno para resultados deterministas y reproducibles.
    
    Args:
        seed: Semilla para generadores aleatorios (default: TRAIN_SEED de config.py)
    
    Returns:
        torch.device configurado
    """
    # 1. Variables de entorno para determinismo
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 2. Fijar seeds de todas las bibliotecas
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 3. Device setup
    device = get_device()
    if device.type == 'cuda':
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    
    # 4. Fijar dtype por defecto
    torch.set_default_dtype(DTYPE)
    
    # 5. Nota: NO usamos torch.use_deterministic_algorithms(True) para mantener
    # compatibilidad con hyper_grid.py, que tampoco lo usa
    
    # 6. Ajustes de backend CUDA/cuDNN
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
    
    # 7. Reducir variabilidad por paralelismo
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    
    print(f"{GREEN}✓ Configuración determinista activada (SEED={seed}){RESET}")
    
    return device


# ==============================================================================
# Directorios de resultados
# ==============================================================================
RESULTS_DIR = "final_results"
RESULTS_DIR_DATA = os.path.join(RESULTS_DIR, "data")
RESULTS_DIR_IMGS = os.path.join(RESULTS_DIR, "images")
os.makedirs(RESULTS_DIR_DATA, exist_ok=True)
os.makedirs(RESULTS_DIR_IMGS, exist_ok=True)


def format_elapsed(sec: float) -> str:
    """Formatea segundos en formato legible (HhMmSs, MmSs o Xs.x)."""
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


def format_filename(p_run: int, concept: str, ext: str = '') -> str:
    """Genera nombre de archivo para resultados de un problema."""
    return os.path.join(RESULTS_DIR_DATA, f"p{p_run}_{concept}{ext}")


def format_img_filename(p_run: int, concept: str, ext: str = '.png') -> str:
    """Genera nombre de archivo de imagen para un problema."""
    return os.path.join(RESULTS_DIR_IMGS, f"p{p_run}_{concept}{ext}")


# ==============================================================================
# Valores exactos de propiedades efectivas (para calcular error relativo)
# ==============================================================================
def effective_exact_all() -> List[float]:
    """Calcula los valores exactos de las 6 propiedades efectivas diagonales."""
    def prom(CC1, CC2):
        return VF_mat1 * CC1 + VF_mat2 * CC2
    
    exacts: List[float] = []
    # p=1: R11_eff
    exacts.append(prom(R11_mat1, R11_mat2) - prom(R13_mat1**2 / R33_mat1, R13_mat2**2 / R33_mat2) 
                  + prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2) ** 2 / prom(1 / R33_mat1, 1 / R33_mat2))
    # p=2: R22_eff
    exacts.append(prom(R22_mat1, R22_mat2) - prom(R23_mat1**2 / R33_mat1, R23_mat2**2 / R33_mat2) 
                  + prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2) ** 2 * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1))
    # p=3: R33_eff
    exacts.append(prom(1 / R33_mat1, 1 / R33_mat2) ** (-1))
    # p=4: R44_eff
    exacts.append(prom(1 / R44_mat1, 1 / R44_mat2) ** (-1))
    # p=5: R55_eff
    exacts.append(prom(1 / R55_mat1, 1 / R55_mat2) ** (-1))
    # p=6: R66_eff
    exacts.append(prom(R66_mat1, R66_mat2))
    return exacts


def effective_exact_cross_terms() -> Dict[str, float]:
    """
    Calcula los valores exactos de los términos cruzados R12_eff, R13_eff, R23_eff.
    
    Para compuestos laminados con dirección de apilamiento x3:
    - R13_eff = <R13/R33> / <1/R33>
    - R23_eff = <R23/R33> / <1/R33>
    - R12_eff = <R12> - <R13*R23/R33> + (<R13/R33> * <R23/R33>) / <1/R33>
    
    donde <·> = VF1·(·)_1 + VF2·(·)_2
    """
    def prom(CC1, CC2):
        return VF_mat1 * CC1 + VF_mat2 * CC2
    
    # Términos auxiliares
    inv_R33_prom = prom(1 / R33_mat1, 1 / R33_mat2)
    R13_over_R33_prom = prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2)
    R23_over_R33_prom = prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2)
    
    # R13_eff = <R13/R33> / <1/R33>
    R13_eff = R13_over_R33_prom / inv_R33_prom
    
    # R23_eff = <R23/R33> / <1/R33>
    R23_eff = R23_over_R33_prom / inv_R33_prom
    
    # R12_eff = <R12> - <R13*R23/R33> + (<R13/R33> * <R23/R33>) / <1/R33>
    R12_eff = (prom(R12_mat1, R12_mat2) 
               - prom(R13_mat1 * R23_mat1 / R33_mat1, R13_mat2 * R23_mat2 / R33_mat2)
               + (R13_over_R33_prom * R23_over_R33_prom) / inv_R33_prom)
    
    return {
        'R12_eff': R12_eff,
        'R13_eff': R13_eff,
        'R23_eff': R23_eff
    }


def extract_cross_terms_from_models(models: Dict[int, PINN], y_points1, y_points2, device) -> Dict[str, float]:
    """
    Extrae las predicciones de términos cruzados R12_eff, R13_eff, R23_eff 
    de los modelos entrenados.
    
    Los términos cruzados se obtienen de:
    - R12_eff: del modelo p=2, componente R1p (i=1, j=2)
    - R13_eff: del modelo p=3, componente R1p (i=1, j=3)
    - R23_eff: del modelo p=3, componente R2p (i=2, j=3)
    
    Args:
        models: Diccionario {p: modelo_entrenado} con al menos p=2 y p=3
        y_points1: Puntos de evaluación en material 1
        y_points2: Puntos de evaluación en material 2
        device: Dispositivo torch (CPU/CUDA)
    
    Returns:
        Diccionario con predicciones {'R12_eff': val, 'R13_eff': val, 'R23_eff': val}
    """
    cross_terms = {}
    
    # R12_eff: del modelo p=2, primer componente (R1p con j=2)
    if 2 in models:
        props = models[2].Effective_Properties(y_points1, y_points2, device)
        cross_terms['R12_eff'] = props[0]  # R1p_eff = R12_eff
    
    # R13_eff y R23_eff: del modelo p=3
    if 3 in models:
        props = models[3].Effective_Properties(y_points1, y_points2, device)
        cross_terms['R13_eff'] = props[0]  # R1p_eff = R13_eff
        cross_terms['R23_eff'] = props[1]  # R2p_eff = R23_eff
    
    return cross_terms


def print_cross_terms_summary(models: Dict[int, PINN], y_points1, y_points2, device):
    """
    Imprime un resumen de los términos cruzados predichos vs exactos.
    """
    # Obtener valores exactos
    exact = effective_exact_cross_terms()
    
    # Obtener predicciones de modelos
    pred = extract_cross_terms_from_models(models, y_points1, y_points2, device)
    
    print(f"\n{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{MAGENTA}            TÉRMINOS CRUZADOS (CROSS TERMS)                    {RESET}")
    print(f"{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════════{RESET}")
    
    terms = ['R12_eff', 'R13_eff', 'R23_eff']
    sources = {'R12_eff': 'modelo p=2', 'R13_eff': 'modelo p=3', 'R23_eff': 'modelo p=3'}
    
    for term in terms:
        exact_val = exact.get(term, float('nan'))
        pred_val = pred.get(term, None)
        source = sources[term]
        
        if pred_val is not None:
            rel_err = abs(pred_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else float('nan')
            status = ICO_OK if rel_err < 1.0 else "⚠️"
            print(f"  {status} {term:8s} (de {source})")
            print(f"       Predicho: {pred_val:15.6f}")
            print(f"       Exacto:   {exact_val:15.6f}")
            print(f"       Error:    {rel_err:15.4f} %")
        else:
            print(f"  ❓ {term:8s}: No disponible (requiere {source})")
    
    print(f"{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════════{RESET}\n")
    
    return pred, exact


# ==============================================================================
# Entrenamiento con ciclos y criterio de parada adaptativo
# ==============================================================================
def run_single_cycle(model: PINN, cycle_num: int, n_adam: int, n_lbfgs: int,
                     log_every: int, progress: bool, callback=None, 
                     callback_every: int = 10, epoch_offset: int = 0,
                     global_start_time: float = None, quiet: bool = False):
    """
    Ejecuta un único ciclo de entrenamiento (Adam + LBFGS).
    
    Args:
        quiet: Si True, suprime mensajes de iteración de Adam y LBFGS.
    
    Returns:
        Número total de épocas ejecutadas en este ciclo
    """
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
    
    # Épocas para este ciclo
    if cycle_num > 0:
        current_n_adam = int(EPOCH_ADAMS_FOLLOWING_CYCLES)
        current_n_lbfgs = int(EPOCH_LBFGS_FOLLOWING_CYCLES)
        current_log_every = max(1, ADAM_LATERCYCLES_PRINT_EVERY)
        try:
            model.adam_print_every = max(1, ADAM_LATERCYCLES_PRINT_EVERY)
            model.lbfgs_print_every = max(1, LBFGS_LATERCYCLES_PRINT_EVERY)
        except Exception:
            pass
    else:
        current_n_adam = int(n_adam)
        current_n_lbfgs = int(n_lbfgs)
        current_log_every = log_every
        try:
            if hasattr(model, 'adam_print_every'):
                delattr(model, 'adam_print_every')
            if hasattr(model, 'lbfgs_print_every'):
                delattr(model, 'lbfgs_print_every')
        except Exception:
            pass
    
    # Callback con offset de épocas
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
        quiet=quiet,
    )
    
    return current_n_adam + current_n_lbfgs


def is_monotonic(values: List[float]) -> bool:
    """
    Verifica si una lista de valores es monótona (creciente o decreciente).
    """
    if len(values) < 2:
        return True
    
    increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
    decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    return increasing or decreasing


def compute_final_reff(reff_history: List[float]) -> tuple:
    """
    Calcula el valor final de R_eff basándose en los últimos CONVERGENCE_WINDOW valores.
    
    Returns:
        (reff_final, method_used)
        method_used: 'last' si es monótono, 'mean' si oscila
    """
    if len(reff_history) < CONVERGENCE_WINDOW:
        return reff_history[-1] if reff_history else float('nan'), 'last'
    
    last_values = reff_history[-CONVERGENCE_WINDOW:]
    
    if is_monotonic(last_values):
        return last_values[-1], 'last'
    else:
        return sum(last_values) / len(last_values), 'mean'


def train_until_convergence(model: PINN, p: int, device: torch.device,
                            n_adam: int, n_lbfgs: int, log_every: int,
                            progress: bool, callback=None, callback_every: int = 10,
                            tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
                            max_cycles: int = 100, quiet: bool = False):
    """
    Entrena el modelo con ciclos (Adam + LBFGS) hasta que R_eff converja.
    
    Criterio de convergencia:
    - A partir del ciclo MIN_CYCLES_BEFORE_CHECK, si la diferencia entre la media
      de R_eff actual y la de hace CONVERGENCE_WINDOW ciclos es menor que tolerance%, se para.
    
    Args:
        model: Modelo PINN a entrenar
        p: Índice del problema (1-6)
        device: Dispositivo de cómputo
        n_adam: Épocas de Adam en el primer ciclo
        n_lbfgs: Épocas de LBFGS en el primer ciclo
        log_every: Frecuencia de logs
        progress: Mostrar barra de progreso
        callback: Función callback(model, epoch)
        callback_every: Frecuencia del callback
        tolerance: Tolerancia de convergencia en porcentaje (default 0.05%)
        max_cycles: Número máximo de ciclos (para evitar bucles infinitos)
        quiet: Si True, suprime mensajes de iteración de Adam y LBFGS.
    
    Returns:
        Dict con información de convergencia:
        {
            'cycles': número de ciclos ejecutados,
            'converged': True si convergió, False si llegó a max_cycles,
            'reff_final': valor final de R_eff,
            'reff_method': 'last' o 'mean',
            'reff_history': lista de R_eff al final de cada ciclo,
            'total_epochs': épocas totales ejecutadas
        }
    """
    epoch_offset = 0
    global_start_time = time.perf_counter()
    
    reff_history = []  # R_eff al final de cada ciclo
    reff_means = []    # Medias de R_eff (a partir del ciclo CONVERGENCE_WINDOW)
    
    converged = False
    cycle = 0
    
    exacts = effective_exact_all()
    reff_exact = exacts[p - 1]
    
    print(f"    {DIM}Iniciando entrenamiento con convergencia adaptativa (tol={tolerance}%){RESET}")
    
    while cycle < max_cycles and not converged:
        # Ejecutar un ciclo
        cycle_epochs = run_single_cycle(
            model, cycle, n_adam, n_lbfgs, log_every, progress,
            callback, callback_every, epoch_offset, global_start_time, quiet
        )
        epoch_offset += cycle_epochs
        cycle += 1
        
        # Evaluar R_eff al final del ciclo
        preds, _, rel_errs = evaluate_model(model, p, device)
        reff_pred = preds[p - 1]
        rel_err = rel_errs[p - 1]
        reff_history.append(reff_pred)
        
        # Calcular media a partir del ciclo CONVERGENCE_WINDOW
        if cycle >= CONVERGENCE_WINDOW:
            current_mean = sum(reff_history[-CONVERGENCE_WINDOW:]) / CONVERGENCE_WINDOW
            reff_means.append(current_mean)
            
            # Log de progreso
            print(f"    {ICO_CYCLE} Ciclo {cycle}: R_eff={reff_pred:.6f}, "
                  f"media({CONVERGENCE_WINDOW})={current_mean:.6f}, err_rel={rel_err:.4f}%")
            
            # Evaluar convergencia a partir del ciclo MIN_CYCLES_BEFORE_CHECK
            if cycle >= MIN_CYCLES_BEFORE_CHECK and len(reff_means) >= CONVERGENCE_WINDOW:
                # Comparar media actual con media de hace CONVERGENCE_WINDOW ciclos
                mean_old = reff_means[-CONVERGENCE_WINDOW]
                mean_new = reff_means[-1]
                
                # Calcular diferencia relativa respecto al valor exacto
                if abs(reff_exact) > 1e-12:
                    diff_pct = abs(mean_new - mean_old) / abs(reff_exact) * 100.0
                else:
                    diff_pct = abs(mean_new - mean_old) * 100.0
                
                print(f"    {DIM}Δmedia = |{mean_new:.6f} - {mean_old:.6f}| = {diff_pct:.6f}% (tol={tolerance}%){RESET}")
                
                if diff_pct < tolerance:
                    converged = True
                    print(f"    {ICO_STOP} Convergencia alcanzada en ciclo {cycle}")
        else:
            # Log básico para ciclos iniciales
            print(f"    {ICO_CYCLE} Ciclo {cycle}: R_eff={reff_pred:.6f}, err_rel={rel_err:.4f}%")
    
    if not converged:
        print(f"    {YELLOW}Máximo de ciclos ({max_cycles}) alcanzado sin convergencia{RESET}")
    
    # Calcular valor final de R_eff
    reff_final, reff_method = compute_final_reff(reff_history)
    
    # Calcular error relativo final
    if abs(reff_exact) > 1e-12:
        final_rel_err = abs(reff_final - reff_exact) / abs(reff_exact) * 100.0
    else:
        final_rel_err = float('nan')
    
    return {
        'cycles': cycle,
        'converged': converged,
        'reff_final': reff_final,
        'reff_method': reff_method,
        'reff_history': reff_history,
        'reff_exact': reff_exact,
        'final_rel_err': final_rel_err,
        'total_epochs': epoch_offset
    }


# ==============================================================================
# Evaluación del modelo
# ==============================================================================
def evaluate_model(model: PINN, p: int, device: torch.device):
    """Evalúa el modelo y calcula propiedades efectivas y errores relativos."""
    y_points1, y_points2 = build_eval_points()
    eff_tuple = model.Effective_Properties(y_points1, y_points2, device)
    preds = [float(x) for x in eff_tuple]
    exacts = effective_exact_all()
    rel_errs = [
        abs(preds[i] - exacts[i]) / abs(exacts[i]) * 100.0 if exacts[i] != 0 else float('nan')
        for i in range(6)
    ]
    return preds, exacts, rel_errs


# ==============================================================================
# Callback para registrar errores durante entrenamiento
# ==============================================================================
def make_relerr_callback(p_run: int, device: torch.device, force_new: bool = True):
    """
    Crea un callback que registra errores relativos durante el entrenamiento.
    
    Returns:
        Tuple de (callback_function, csv_path)
    """
    rel_csv = format_filename(p_run, 'relerr', ext='.csv')
    
    if force_new or not os.path.exists(rel_csv):
        with open(rel_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            header = ['epoch', f'rel_err_p{p_run}_pct', f'R_eff_pred_p{p_run}', f'R_eff_exact_p{p_run}',
                      'loss_total', 'loss_collocation', 'loss_uniqueness_condition', 'loss_interface']
            w.writerow(header)
    
    def _cb(model: PINN, epoch: int):
        try:
            preds, exacts, rel_errs = evaluate_model(model, p_run, device)
            rel_val = float(rel_errs[p_run - 1]) if rel_errs and len(rel_errs) >= p_run else float('nan')
            reff_pred = float(preds[p_run - 1]) if preds and len(preds) >= p_run else float('nan')
            reff_exact = float(exacts[p_run - 1]) if exacts and len(exacts) >= p_run else float('nan')
            
            # Obtener componentes de pérdida
            try:
                lt, lc, lu, li = model.loss()
                lt_f = float(lt.detach().cpu().item())
                lc_f = float(lc.detach().cpu().item())
                lu_f = float(lu.detach().cpu().item())
                li_f = float(li.detach().cpu().item())
            except Exception:
                lt_f = float(model.loss_log[-1]) if getattr(model, 'loss_log', None) else float('nan')
                lc_f = float('nan')
                lu_f = float('nan')
                li_f = float('nan')
            
            # Aplicar pesos adaptativos si existen
            try:
                wc = float(model.w_collocation.detach().cpu().item())
                wu = float(model.w_uniqueness.detach().cpu().item())
                wi = float(model.w_interface.detach().cpu().item())
                if not math.isnan(lc_f):
                    lc_f *= wc
                if not math.isnan(lu_f):
                    lu_f *= wu
                if not math.isnan(li_f):
                    li_f *= wi
            except Exception:
                pass
            
            with open(rel_csv, 'a', newline='') as fh:
                w = csv.writer(fh)
                w.writerow([epoch, f"{rel_val:.6f}", f"{reff_pred:.6f}", f"{reff_exact:.6f}",
                           f"{lt_f:.12e}", f"{lc_f:.12e}", f"{lu_f:.12e}", f"{li_f:.12e}"])
        except Exception:
            pass
    
    return _cb, rel_csv


# ==============================================================================
# Escritura de CSVs de resultados
# ==============================================================================
def write_loss_csv(p_run: int, model: PINN):
    """Escribe CSV con pérdidas por época."""
    loss_csv = format_filename(p_run, 'loss', ext='.csv')
    
    try:
        with open(loss_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'loss_total', 'loss_collocation', 'loss_uniqueness_condition', 'loss_interface',
                       'weighted_collocation', 'weighted_uniqueness', 'weighted_interface'])
            
            n_epochs = len(getattr(model, 'loss_log', []) or [])
            weighted_coll = getattr(model, 'weighted_collocation_log', [])
            weighted_uniq = getattr(model, 'weighted_uniqueness_log', [])
            weighted_intf = getattr(model, 'weighted_interface_log', [])
            
            for idx in range(n_epochs):
                ep = idx + 1
                lt = float(model.loss_log[idx]) if idx < len(model.loss_log) else float('nan')
                lc = float(model.loss_collocation_log[idx]) if hasattr(model, 'loss_collocation_log') and idx < len(model.loss_collocation_log) else float('nan')
                lu = float(model.loss_uniqueness_condition_log[idx]) if hasattr(model, 'loss_uniqueness_condition_log') and idx < len(model.loss_uniqueness_condition_log) else float('nan')
                li = float(model.loss_interface_log[idx]) if hasattr(model, 'loss_interface_log') and idx < len(model.loss_interface_log) else float('nan')
                
                wc_contrib = float(weighted_coll[idx]) if idx < len(weighted_coll) else float('nan')
                wu_contrib = float(weighted_uniq[idx]) if idx < len(weighted_uniq) else float('nan')
                wi_contrib = float(weighted_intf[idx]) if idx < len(weighted_intf) else float('nan')
                
                w.writerow([ep, f"{lt:.12e}", f"{lc:.12e}", f"{lu:.12e}", f"{li:.12e}",
                           f"{wc_contrib:.12e}", f"{wu_contrib:.12e}", f"{wi_contrib:.12e}"])
        return loss_csv
    except Exception:
        return None


# ==============================================================================
# Generación de gráficas
# ==============================================================================
def generate_plots(p_run: int, loss_csv: str, relerr_csv: str, cycles: int = 1):
    """Genera gráficas de pérdida y error relativo para un problema."""
    
    # Calcular transiciones entre fases para marcar en las gráficas
    transitions = []
    try:
        cur = 0
        for c in range(1, cycles + 1):
            if c == 1:
                a = int(EPOCH_ADAMS)
                b = int(EPOCH_LBFGS)
            else:
                a = int(EPOCH_ADAMS_FOLLOWING_CYCLES)
                b = int(EPOCH_LBFGS_FOLLOWING_CYCLES)
            cur += a
            transitions.append((cur, 'adam->lbfgs'))
            cur += b
            transitions.append((cur, 'lbfgs->adam'))
    except Exception:
        transitions = []
    
    # Gráfica de pérdida total
    if loss_csv and os.path.exists(loss_csv):
        try:
            plot_path = format_img_filename(p_run, 'loss')
            plot_utils.plot_loss_overlays([loss_csv], [f'p{p_run}'], plot_path, smooth_alpha=None)
            print(f"{MAGENTA}{ICO_SAVE} Gráfica: {plot_path}{RESET}")
            
            # Versión suavizada
            plot_path_s = format_img_filename(p_run, 'loss_smooth')
            plot_utils.plot_loss_overlays([loss_csv], [f'p{p_run}'], plot_path_s, smooth_alpha=0.12)
        except Exception as e:
            print(f"{YELLOW}Error generando gráfica de pérdida: {e}{RESET}")
        
        # Gráfica de componentes
        try:
            plot_path = format_img_filename(p_run, 'loss_components')
            plot_utils.plot_loss_components(loss_csv, plot_path, smooth_alpha=None)
            print(f"{MAGENTA}{ICO_SAVE} Gráfica: {plot_path}{RESET}")
            
            plot_path_s = format_img_filename(p_run, 'loss_components_smooth')
            plot_utils.plot_loss_components(loss_csv, plot_path_s, smooth_alpha=0.12)
        except Exception:
            pass
    
    # Gráfica de error relativo
    if relerr_csv and os.path.exists(relerr_csv):
        try:
            xs, ys, reff_preds, reff_exact = [], [], [], None
            
            with open(relerr_csv, 'r', newline='') as fh:
                r = csv.reader(fh)
                headers = next(r)
                idx_epoch = 0
                idx_rel = None
                idx_reff_pred = None
                idx_reff_exact = None
                
                for i, h in enumerate(headers):
                    if 'rel_err' in h.lower():
                        idx_rel = i
                    if h.startswith('R_eff_pred'):
                        idx_reff_pred = i
                    if h.startswith('R_eff_exact'):
                        idx_reff_exact = i
                
                if idx_rel is None:
                    idx_rel = 1
                
                for row in r:
                    if not row:
                        continue
                    try:
                        ep = int(row[idx_epoch])
                        rel = float(row[idx_rel])
                        xs.append(ep)
                        ys.append(rel)
                        
                        if idx_reff_pred is not None and idx_reff_pred < len(row):
                            reff_preds.append(float(row[idx_reff_pred]))
                        if idx_reff_exact is not None and reff_exact is None and idx_reff_exact < len(row):
                            reff_exact = float(row[idx_reff_exact])
                    except Exception:
                        continue
            
            if xs and ys:
                plot_path = format_img_filename(p_run, 'relerr')
                plot_utils.plot_relerr(xs, ys, 'Epoch', f"Error Relativo p{p_run}", plot_path, logx=False, smooth_alpha=None)
                print(f"{MAGENTA}{ICO_SAVE} Gráfica: {plot_path}{RESET}")
                
                plot_path_s = format_img_filename(p_run, 'relerr_smooth')
                plot_utils.plot_relerr(xs, ys, 'Epoch', f"Error Relativo p{p_run}", plot_path_s, logx=False, smooth_alpha=0.12)
            
            # Gráfica de evolución de R_eff
            if reff_preds and reff_exact is not None:
                try:
                    plot_path = format_img_filename(p_run, 'reff_evolution')
                    plot_utils.plot_reff_evolution(xs, reff_preds, reff_exact, p_run, plot_path, smooth_alpha=None, transitions=transitions)
                    print(f"{MAGENTA}{ICO_SAVE} Gráfica: {plot_path}{RESET}")
                    
                    plot_path_s = format_img_filename(p_run, 'reff_evolution_smooth')
                    plot_utils.plot_reff_evolution(xs, reff_preds, reff_exact, p_run, plot_path_s, smooth_alpha=0.12)
                except Exception:
                    pass
        except Exception as e:
            print(f"{YELLOW}Error generando gráfica de error relativo: {e}{RESET}")


# ==============================================================================
# Entrenamiento de un problema
# ==============================================================================
def train_problem(p: int, device: torch.device, tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
                  max_cycles: int = 100, force_retrain: bool = False, quiet: bool = False) -> Dict[str, Any]:
    """
    Entrena un problema específico con criterio de parada adaptativo.
    
    Args:
        p: Índice del problema (1-6)
        device: Dispositivo de cómputo
        tolerance: Tolerancia de convergencia en porcentaje (default 0.05%)
        max_cycles: Número máximo de ciclos (default 100)
        force_retrain: Forzar reentrenamiento aunque exista modelo
        quiet: Si True, suprime mensajes de iteración de Adam y LBFGS
    
    Returns:
        Dict con resultados del entrenamiento
    """
    print(f"\n{BOLD}{CYAN}{ICO_TRAIN} Entrenando problema p={p}{RESET}")
    print(f"    Configuración: Adam={EPOCH_ADAMS} + {EPOCH_ADAMS_FOLLOWING_CYCLES}/ciclo, "
          f"LBFGS={EPOCH_LBFGS} + {EPOCH_LBFGS_FOLLOWING_CYCLES}/ciclo")
    print(f"    LR={ADAM_LR:.1e}, Arquitectura: {LAYERS}")
    print(f"    Criterio de parada: tolerancia={tolerance}%, max_ciclos={max_cycles}")
    
    # Verificar si existe un modelo guardado
    model_path = format_filename(p, 'model', ext='.pt')
    
    # Construir datos de entrenamiento
    y_coll_mat1, y_coll_mat2, y_interface, y_uniq1, y_uniq2 = build_train_data()
    PARAM = {'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}
    
    # Crear modelo
    model = PINN(y_coll_mat1, y_coll_mat2, y_interface, y_uniq1, y_uniq2,
                 param=PARAM, layers=LAYERS, device=device, prob_ind=p)
    
    # Cargar modelo existente si no forzamos reentrenamiento
    if os.path.exists(model_path) and not force_retrain:
        try:
            sd = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(sd)
            print(f"    {GREEN}Modelo cargado desde {model_path}{RESET}")
            
            # Evaluar modelo cargado
            preds, exacts, rel_errs = evaluate_model(model, p, device)
            final_loss = model.loss_log[-1] if getattr(model, 'loss_log', None) else float('nan')
            
            return {
                'p': p,
                'time_sec': 0.0,
                'cycles': 0,
                'converged': True,
                'final_loss': final_loss,
                'preds': preds,
                'exacts': exacts,
                'rel_errs': rel_errs,
                'reff_final': preds[p-1],
                'reff_exact': exacts[p-1],
                'reff_method': 'loaded',
                'model': model,
                'loaded': True
            }
        except Exception as e:
            print(f"    {YELLOW}Error cargando modelo: {e}. Reentrenando...{RESET}")
    
    # Configurar learning rate
    for g in model.optimizer_Adam.param_groups:
        g['lr'] = ADAM_LR
    model._last_adam_lr = ADAM_LR
    
    # Crear callback para registrar errores
    relerr_cb, relerr_csv = make_relerr_callback(p, device, force_new=True)
    
    # Entrenar hasta convergencia
    start_time = time.perf_counter()
    conv_info = train_until_convergence(
        model, p, device,
        n_adam=EPOCH_ADAMS, n_lbfgs=EPOCH_LBFGS,
        log_every=max(1, ADAM_PRINT_EVERY),
        progress=False,
        callback=relerr_cb,
        callback_every=10,
        tolerance=tolerance,
        max_cycles=max_cycles,
        quiet=quiet
    )
    training_time = time.perf_counter() - start_time
    
    # Obtener pérdida final
    final_loss = model.loss_log[-1] if model.loss_log else float('nan')
    
    # Guardar modelo
    try:
        torch.save(model.state_dict(), model_path)
        print(f"    {MAGENTA}{ICO_SAVE} Modelo guardado: {model_path}{RESET}")
    except Exception as e:
        print(f"    {YELLOW}Error guardando modelo: {e}{RESET}")
    
    # Escribir CSV de pérdidas
    loss_csv = write_loss_csv(p, model)
    
    # Guardar historial de R_eff por ciclo
    reff_history_csv = format_filename(p, 'reff_history', ext='.csv')
    try:
        with open(reff_history_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['cycle', 'R_eff_pred', 'R_eff_exact', 'rel_err_pct'])
            for i, reff in enumerate(conv_info['reff_history'], start=1):
                rel_err = abs(reff - conv_info['reff_exact']) / abs(conv_info['reff_exact']) * 100.0
                w.writerow([i, f"{reff:.6f}", f"{conv_info['reff_exact']:.6f}", f"{rel_err:.4f}"])
        print(f"    {MAGENTA}{ICO_SAVE} Historial R_eff: {reff_history_csv}{RESET}")
    except Exception:
        pass
    
    # Generar gráficas
    generate_plots(p, loss_csv, relerr_csv, conv_info['cycles'])
    
    # Mostrar resultados
    method_str = "último valor" if conv_info['reff_method'] == 'last' else f"media de últimos {CONVERGENCE_WINDOW}"
    convergence_str = "✅ Convergido" if conv_info['converged'] else "⚠️ Max ciclos"
    
    print(f"\n    {GREEN}{ICO_OK} Problema p={p} completado{RESET}")
    print(f"    {convergence_str} en {conv_info['cycles']} ciclos ({format_elapsed(training_time)})")
    print(f"    Épocas totales: {conv_info['total_epochs']}")
    print(f"    Pérdida final: {final_loss:.6e}")
    print(f"    R_eff final ({method_str}): {conv_info['reff_final']:.6f}")
    print(f"    R_eff exacto: {conv_info['reff_exact']:.6f}")
    print(f"    Error relativo final: {conv_info['final_rel_err']:.4f}%")
    
    return {
        'p': p,
        'time_sec': training_time,
        'cycles': conv_info['cycles'],
        'converged': conv_info['converged'],
        'final_loss': final_loss,
        'preds': [conv_info['reff_final']] * 6,  # Solo p relevante
        'exacts': effective_exact_all(),
        'rel_errs': [conv_info['final_rel_err']] * 6,  # Solo p relevante
        'reff_final': conv_info['reff_final'],
        'reff_exact': conv_info['reff_exact'],
        'reff_method': conv_info['reff_method'],
        'reff_history': conv_info['reff_history'],
        'total_epochs': conv_info['total_epochs'],
        'model': model,
        'loaded': False
    }


# ==============================================================================
# Funciones de estadísticas para multi-run
# ==============================================================================
def truncated_mean(values: List[float], trim: int = 1) -> float:
    """
    Calcula la media truncada quitando los valores extremos.
    
    Args:
        values: Lista de valores numéricos
        trim: Número de valores a quitar por cada lado (default: 1 = quita 1 max y 1 min)
    
    Returns:
        Media truncada, o NaN si no hay suficientes valores
    """
    if len(values) < 2 * trim + 1:
        # No hay suficientes valores para truncar
        return statistics.mean(values) if values else float('nan')
    
    sorted_vals = sorted(values)
    trimmed = sorted_vals[trim:-trim] if trim > 0 else sorted_vals
    return statistics.mean(trimmed) if trimmed else float('nan')


def compute_statistics(values: List[float], truncate_pct: float = DEFAULT_TRUNCATE_PCT) -> Dict[str, float]:
    """
    Calcula estadísticas para una lista de valores: media, media truncada y mediana.
    
    Args:
        values: Lista de valores R_eff de diferentes runs
        truncate_pct: Porcentaje de valores a quitar por cada lado para media truncada.
                      Por ejemplo, 20 = 20%, si N=20 se quitan 4 arriba y 4 abajo.
    
    Returns:
        Dict con 'mean', 'truncated_mean', 'median', 'std', 'min', 'max', 'trim_count'
    """
    if not values:
        return {k: float('nan') for k in ['mean', 'truncated_mean', 'median', 'std', 'min', 'max', 'trim_count']}
    
    clean_values = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    
    if not clean_values:
        return {k: float('nan') for k in ['mean', 'truncated_mean', 'median', 'std', 'min', 'max', 'trim_count']}
    
    # Calcular trim_count a partir del porcentaje
    # Por ejemplo, si truncate_pct=20 y N=20, trim_count=4 (quita 4 arriba y 4 abajo)
    trim_count = max(0, int(len(clean_values) * truncate_pct / 100.0))
    
    return {
        'mean': statistics.mean(clean_values),
        'truncated_mean': truncated_mean(clean_values, trim_count),
        'median': statistics.median(clean_values),
        'std': statistics.stdev(clean_values) if len(clean_values) > 1 else 0.0,
        'min': min(clean_values),
        'max': max(clean_values),
        'trim_count': trim_count
    }


def run_single_iteration(iteration: int, seed: int, targets: List[int], 
                         tolerance: float, max_cycles: int, 
                         base_results_dir: str, quiet: bool = False) -> Dict[str, Any]:
    """
    Ejecuta una iteración completa de entrenamiento con un seed específico.
    
    Args:
        iteration: Número de iteración (1-indexed)
        seed: Semilla para reproducibilidad
        targets: Lista de problemas a entrenar (1-6)
        tolerance: Tolerancia de convergencia
        max_cycles: Máximo de ciclos
        base_results_dir: Directorio base para resultados de esta iteración
        quiet: Si True, suprime mensajes de iteración de Adam y LBFGS
    
    Returns:
        Dict con resultados de todos los problemas y términos cruzados
    """
    global RESULTS_DIR, RESULTS_DIR_DATA, RESULTS_DIR_IMGS
    
    # Configurar directorios para esta iteración
    RESULTS_DIR = os.path.join(base_results_dir, f"run_{iteration:02d}_seed_{seed}")
    RESULTS_DIR_DATA = os.path.join(RESULTS_DIR, "data")
    RESULTS_DIR_IMGS = os.path.join(RESULTS_DIR, "images")
    os.makedirs(RESULTS_DIR_DATA, exist_ok=True)
    os.makedirs(RESULTS_DIR_IMGS, exist_ok=True)
    
    print(f"\n{BOLD}{BLUE}{'='*90}{RESET}")
    print(f"{BOLD}{BLUE}  ITERACIÓN {iteration}: SEED = {seed}{RESET}")
    print(f"{BOLD}{BLUE}{'='*90}{RESET}")
    
    # Configurar determinismo con este seed
    device = setup_deterministic(seed)
    
    results = []
    
    # Entrenar cada problema
    for p in sorted(targets):
        result = train_problem(p, device, tolerance=tolerance, 
                               max_cycles=max_cycles, force_retrain=True, quiet=quiet)  # Siempre forzar en multi-run
        results.append(result)
    
    # Recopilar modelos y Reff
    trained_models = {r['p']: r['model'] for r in results if 'model' in r}
    reff_values = {}
    
    for r in results:
        p = r['p']
        reff_values[f'R{p}{p}_eff'] = r.get('reff_final', float('nan'))
    
    # Obtener términos cruzados si están disponibles los modelos necesarios
    cross_terms = {}
    if 2 in trained_models and 3 in trained_models:
        y_points1, y_points2 = build_eval_points()
        cross_pred = extract_cross_terms_from_models(trained_models, y_points1, y_points2, device)
        cross_terms = cross_pred
    elif 2 in trained_models:
        y_points1, y_points2 = build_eval_points()
        cross_pred = extract_cross_terms_from_models(trained_models, y_points1, y_points2, device)
        if 'R12_eff' in cross_pred:
            cross_terms['R12_eff'] = cross_pred['R12_eff']
    elif 3 in trained_models:
        y_points1, y_points2 = build_eval_points()
        cross_pred = extract_cross_terms_from_models(trained_models, y_points1, y_points2, device)
        for term in ['R13_eff', 'R23_eff']:
            if term in cross_pred:
                cross_terms[term] = cross_pred[term]
    
    return {
        'iteration': iteration,
        'seed': seed,
        'results': results,
        'reff_values': reff_values,
        'cross_terms': cross_terms
    }


def run_multi_seed_analysis(n_runs: int, targets: List[int], tolerance: float,
                            max_cycles: int, base_seed: int = TRAIN_SEED,
                            quiet: bool = False, truncate_pct: float = DEFAULT_TRUNCATE_PCT) -> Dict[str, Any]:
    """
    Ejecuta N iteraciones del entrenamiento completo con diferentes seeds.
    
    Args:
        n_runs: Número de iteraciones a ejecutar
        targets: Lista de problemas a entrenar (1-6)
        tolerance: Tolerancia de convergencia
        max_cycles: Máximo de ciclos por problema
        base_seed: Seed base para generar los n_runs seeds
        quiet: Si True, suprime mensajes de iteración de Adam y LBFGS
        truncate_pct: Porcentaje de valores a quitar por cada lado para media truncada
    
    Returns:
        Dict con todos los resultados y estadísticas consolidadas
    """
    global RESULTS_DIR, RESULTS_DIR_DATA, RESULTS_DIR_IMGS
    
    # Directorio base para multi-run
    base_results_dir = "final_results_multirun"
    os.makedirs(base_results_dir, exist_ok=True)
    
    print(f"\n{BOLD}{MAGENTA}{'='*90}{RESET}")
    print(f"{BOLD}{MAGENTA}  ANÁLISIS MULTI-SEED: {n_runs} ITERACIONES{RESET}")
    print(f"{BOLD}{MAGENTA}{'='*90}{RESET}")
    print(f"Problemas a entrenar: {targets}")
    print(f"Tolerancia de convergencia: {tolerance}%")
    print(f"Máximo de ciclos: {max_cycles}")
    print(f"Directorio base: {base_results_dir}")
    
    # Generar los N seeds de forma determinista
    np.random.seed(base_seed)
    seeds = [base_seed + i * 1000 for i in range(n_runs)]  # Seeds separados y predecibles
    print(f"Seeds a usar: {seeds}")
    
    global_start = time.perf_counter()
    all_runs = []
    
    # Ejecutar cada iteración
    for i, seed in enumerate(seeds, start=1):
        run_result = run_single_iteration(i, seed, targets, tolerance, max_cycles, base_results_dir, quiet)
        all_runs.append(run_result)
    
    total_time = time.perf_counter() - global_start
    
    # Restaurar directorio de resultados para el resumen final
    RESULTS_DIR = base_results_dir
    RESULTS_DIR_DATA = os.path.join(RESULTS_DIR, "data_summary")
    RESULTS_DIR_IMGS = os.path.join(RESULTS_DIR, "images_summary")
    os.makedirs(RESULTS_DIR_DATA, exist_ok=True)
    os.makedirs(RESULTS_DIR_IMGS, exist_ok=True)
    
    # Consolidar resultados por propiedad
    all_reff = {}  # {property_name: [valores de cada run]}
    
    # Propiedades diagonales
    for p in targets:
        key = f'R{p}{p}_eff'
        all_reff[key] = []
        for run in all_runs:
            if key in run['reff_values']:
                all_reff[key].append(run['reff_values'][key])
    
    # Términos cruzados
    cross_keys = ['R12_eff', 'R13_eff', 'R23_eff']
    for key in cross_keys:
        all_reff[key] = []
        for run in all_runs:
            if key in run['cross_terms']:
                all_reff[key].append(run['cross_terms'][key])
    
    # Calcular estadísticas para cada propiedad
    stats = {}
    for key, values in all_reff.items():
        if values:
            stats[key] = compute_statistics(values, truncate_pct)
            stats[key]['n_values'] = len(values)
            stats[key]['all_values'] = values
    
    # Obtener valores exactos para calcular errores
    exacts_diag = effective_exact_all()
    exacts_cross = effective_exact_cross_terms()
    exact_values = {}
    for p in targets:
        exact_values[f'R{p}{p}_eff'] = exacts_diag[p-1]
    for key in cross_keys:
        exact_values[key] = exacts_cross[key]
    
    # Calcular errores relativos para las estadísticas
    for key in stats:
        if key in exact_values and exact_values[key] != 0:
            exact_val = exact_values[key]
            stats[key]['error_mean'] = abs(stats[key]['mean'] - exact_val) / abs(exact_val) * 100
            stats[key]['error_truncated'] = abs(stats[key]['truncated_mean'] - exact_val) / abs(exact_val) * 100
            stats[key]['error_median'] = abs(stats[key]['median'] - exact_val) / abs(exact_val) * 100
            stats[key]['exact'] = exact_val
    
    # Escribir CSV de resumen
    write_multirun_summary_csv(stats, n_runs, base_results_dir, truncate_pct)
    
    # Escribir CSV detallado con todos los valores
    write_multirun_detailed_csv(all_runs, base_results_dir)
    
    # Guardar todos los Reff para cálculos posteriores
    save_all_reff_values(all_reff, base_results_dir, truncate_pct)
    
    # Mostrar resumen final
    print_multirun_summary(stats, n_runs, total_time, truncate_pct)
    
    return {
        'n_runs': n_runs,
        'seeds': seeds,
        'all_runs': all_runs,
        'stats': stats,
        'total_time': total_time
    }


def write_multirun_summary_csv(stats: Dict[str, Dict], n_runs: int, base_dir: str, truncate_pct: float = DEFAULT_TRUNCATE_PCT):
    """Escribe CSV con estadísticas resumidas del multi-run."""
    csv_path = os.path.join(base_dir, "data_summary", "multirun_statistics.csv")
    
    headers = ['property', 'n_samples', 'mean', 'truncated_mean', 'median', 'std', 'min', 'max',
               'exact', 'error_mean_%', 'error_truncated_%', 'error_median_%', 'truncate_pct', 'trim_count']
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        
        for prop in sorted(stats.keys()):
            s = stats[prop]
            row = [
                prop,
                s.get('n_values', 0),
                f"{s['mean']:.8f}",
                f"{s['truncated_mean']:.8f}",
                f"{s['median']:.8f}",
                f"{s['std']:.8f}",
                f"{s['min']:.8f}",
                f"{s['max']:.8f}",
                f"{s.get('exact', float('nan')):.8f}",
                f"{s.get('error_mean', float('nan')):.6f}",
                f"{s.get('error_truncated', float('nan')):.6f}",
                f"{s.get('error_median', float('nan')):.6f}",
                f"{truncate_pct:.1f}",
                s.get('trim_count', 0)
            ]
            w.writerow(row)
    
    print(f"{MAGENTA}{ICO_SAVE} Estadísticas multi-run: {csv_path}{RESET}")


def write_multirun_detailed_csv(all_runs: List[Dict], base_dir: str):
    """Escribe CSV detallado con todos los valores de cada run."""
    csv_path = os.path.join(base_dir, "data_summary", "multirun_detailed.csv")
    
    # Determinar todas las propiedades presentes
    all_props = set()
    for run in all_runs:
        all_props.update(run['reff_values'].keys())
        all_props.update(run['cross_terms'].keys())
    
    sorted_props = sorted(all_props)
    headers = ['run', 'seed'] + sorted_props
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        
        for run in all_runs:
            row = [run['iteration'], run['seed']]
            for prop in sorted_props:
                if prop in run['reff_values']:
                    row.append(f"{run['reff_values'][prop]:.8f}")
                elif prop in run['cross_terms']:
                    row.append(f"{run['cross_terms'][prop]:.8f}")
                else:
                    row.append('')
            w.writerow(row)
    
    print(f"{MAGENTA}{ICO_SAVE} Detalle multi-run: {csv_path}{RESET}")


def save_all_reff_values(all_reff: Dict[str, List[float]], base_dir: str, truncate_pct: float = DEFAULT_TRUNCATE_PCT):
    """
    Guarda todos los valores de Reff de cada run en un archivo para cálculos posteriores.
    
    Args:
        all_reff: Diccionario {property_name: [valores de cada run]}
        base_dir: Directorio base para resultados
        truncate_pct: Porcentaje de truncamiento utilizado
    """
    csv_path = os.path.join(base_dir, "data_summary", "all_reff_values.csv")
    
    # Obtener todas las propiedades
    all_props = sorted(all_reff.keys())
    
    # Encontrar el máximo número de runs
    max_runs = max(len(values) for values in all_reff.values()) if all_reff else 0
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        
        # Escribir metadata como comentario
        w.writerow([f"# truncate_pct={truncate_pct}"])
        
        # Header con propiedades como columnas
        w.writerow(['run'] + all_props)
        
        # Una fila por run
        for i in range(max_runs):
            row = [i + 1]  # Número de run (1-indexed)
            for prop in all_props:
                values = all_reff.get(prop, [])
                if i < len(values):
                    row.append(f"{values[i]:.10f}")  # Alta precisión para cálculos posteriores
                else:
                    row.append('')
            w.writerow(row)
    
    print(f"{MAGENTA}{ICO_SAVE} Todos los Reff guardados: {csv_path}{RESET}")
    
    # También guardar en formato JSON para fácil lectura programática
    import json
    json_path = os.path.join(base_dir, "data_summary", "all_reff_values.json")
    json_data = {
        'truncate_pct': truncate_pct,
        'n_runs': max_runs,
        'reff_values': {k: [float(v) for v in vals] for k, vals in all_reff.items()},
        'metadata': {
            'description': 'Valores de R_eff obtenidos en cada run para cálculos posteriores',
            'format': 'property_name: [list of values, one per run]'
        }
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"{MAGENTA}{ICO_SAVE} Todos los Reff (JSON): {json_path}{RESET}")


def print_multirun_summary(stats: Dict[str, Dict], n_runs: int, total_time: float, truncate_pct: float = DEFAULT_TRUNCATE_PCT):
    """Imprime resumen formateado del análisis multi-run."""
    print(f"\n{BOLD}{GREEN}{'='*100}{RESET}")
    print(f"{BOLD}{GREEN}  RESUMEN MULTI-RUN ({n_runs} ITERACIONES){RESET}")
    print(f"{BOLD}{GREEN}{'='*100}{RESET}")
    print(f"Tiempo total: {format_elapsed(total_time)}")
    # Mostrar el porcentaje de truncamiento y el trim_count de la primera propiedad
    first_trim = next((s.get('trim_count', 0) for s in stats.values()), 0)
    print(f"Media truncada: {truncate_pct}% por cada lado ({first_trim} valores eliminados por lado)")
    print()
    
    # Tabla de resultados
    header = f"{'Propiedad':<12} {'N':<4} {'Media':<14} {'MediaTrunc':<14} {'Mediana':<14} {'Std':<10} {'Exacto':<14} {'Err Mean%':<10} {'Err Trunc%':<10} {'Err Med%':<10}"
    print(header)
    print("-" * 120)
    
    # Primero propiedades diagonales
    diag_props = [k for k in sorted(stats.keys()) if k.startswith('R') and k[1] == k[2]]
    for prop in diag_props:
        s = stats[prop]
        print(f"{prop:<12} {s.get('n_values', 0):<4} {s['mean']:<14.8f} {s['truncated_mean']:<14.8f} "
              f"{s['median']:<14.8f} {s['std']:<10.6f} {s.get('exact', float('nan')):<14.8f} "
              f"{s.get('error_mean', float('nan')):<10.4f} {s.get('error_truncated', float('nan')):<10.4f} "
              f"{s.get('error_median', float('nan')):<10.4f}")
    
    # Luego términos cruzados
    cross_props = [k for k in sorted(stats.keys()) if k not in diag_props]
    if cross_props:
        print()
        print(f"{BOLD}Términos cruzados:{RESET}")
        for prop in cross_props:
            s = stats[prop]
            print(f"{prop:<12} {s.get('n_values', 0):<4} {s['mean']:<14.8f} {s['truncated_mean']:<14.8f} "
                  f"{s['median']:<14.8f} {s['std']:<10.6f} {s.get('exact', float('nan')):<14.8f} "
                  f"{s.get('error_mean', float('nan')):<10.4f} {s.get('error_truncated', float('nan')):<10.4f} "
                  f"{s.get('error_median', float('nan')):<10.4f}")
    
    print(f"\n{GREEN}{ICO_OK} Análisis multi-run completado{RESET}")


# ==============================================================================
# Función principal
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PINN Final - Entrenamiento con criterio de parada adaptativo"
    )
    parser.add_argument('--targets', type=str, default='1,2,3,4,5,6',
                        help='Problemas a entrenar, separados por coma (default: 1,2,3,4,5,6)')
    parser.add_argument('--target', type=int, default=None, choices=range(1,7),
                        help='Problema único a entrenar (anula --targets)')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_CONVERGENCE_TOLERANCE,
                        help=f'Tolerancia de convergencia en %% (default: {DEFAULT_CONVERGENCE_TOLERANCE}%%)')
    parser.add_argument('--max-cycles', type=int, default=100,
                        help='Número máximo de ciclos antes de parar (default: 100)')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Forzar reentrenamiento aunque existan modelos guardados')
    parser.add_argument('--n-runs', type=int, default=1,
                        help=f'Número de iteraciones con diferentes seeds (default: 1, usar {DEFAULT_N_RUNS} para análisis robusto)')
    parser.add_argument('--base-seed', type=int, default=TRAIN_SEED,
                        help=f'Seed base para generar los seeds de cada iteración (default: {TRAIN_SEED})')
    parser.add_argument('--quiet-iterations', action='store_true',
                        help='Silencia los mensajes de iteración de Adam/LBFGS dentro de los ciclos')
    parser.add_argument('--truncate-pct', type=float, default=DEFAULT_TRUNCATE_PCT,
                        help=f'Porcentaje de valores a eliminar por cada lado para media truncada (default: {DEFAULT_TRUNCATE_PCT}%%)')
    args = parser.parse_args()
    
    # Determinar problemas a entrenar
    if args.target is not None:
        targets = [args.target]
    else:
        targets = [int(x) for x in args.targets.split(',') if x.strip()]
    
    # Si n_runs > 1, ejecutar análisis multi-seed
    if args.n_runs > 1:
        run_multi_seed_analysis(
            n_runs=args.n_runs,
            targets=targets,
            tolerance=args.tolerance,
            max_cycles=args.max_cycles,
            base_seed=args.base_seed,
            quiet=args.quiet_iterations,
            truncate_pct=args.truncate_pct
        )
        return
    
    # Configurar determinismo y obtener dispositivo
    device = setup_deterministic(TRAIN_SEED)
    print(f"{BOLD}{CYAN}{ICO_RUN} PINN Final - Entrenamiento con Convergencia Adaptativa{RESET}")
    print(f"Dispositivo: {device}, Dtype: {DTYPE}")
    print(f"Directorio de resultados: {RESULTS_DIR}")
    
    # Mostrar configuración
    print(f"\n{BOLD}Configuración de entrenamiento:{RESET}")
    print(f"  Épocas Adam (ciclo 1): {EPOCH_ADAMS}")
    print(f"  Épocas LBFGS (ciclo 1): {EPOCH_LBFGS}")
    print(f"  Épocas Adam (ciclos siguientes): {EPOCH_ADAMS_FOLLOWING_CYCLES}")
    print(f"  Épocas LBFGS (ciclos siguientes): {EPOCH_LBFGS_FOLLOWING_CYCLES}")
    print(f"  Learning rate Adam: {ADAM_LR:.1e}")
    print(f"  Arquitectura: {LAYERS}")
    print(f"  N_COLLOCATION: {N_COLLOCATION}")
    
    print(f"\n{BOLD}Criterio de parada:{RESET}")
    print(f"  Tolerancia de convergencia: {args.tolerance}%")
    print(f"  Mínimo ciclos antes de evaluar: {MIN_CYCLES_BEFORE_CHECK}")
    print(f"  Ventana de comparación: {CONVERGENCE_WINDOW} ciclos")
    print(f"  Máximo de ciclos: {args.max_cycles}")
    
    print(f"\n{BOLD}Problemas a entrenar:{RESET} {targets}")
    
    global_start = time.perf_counter()
    results = []
    
    # Entrenar cada problema
    for p in sorted(targets):
        result = train_problem(p, device, tolerance=args.tolerance, 
                               max_cycles=args.max_cycles, force_retrain=args.force_retrain,
                               quiet=args.quiet_iterations)
        results.append(result)
    
    total_time = time.perf_counter() - global_start
    
    # Recopilar modelos entrenados
    trained_models = {r['p']: r['model'] for r in results if 'model' in r}
    
    # Mostrar términos cruzados si tenemos los modelos necesarios (p=2 y p=3)
    if 2 in trained_models and 3 in trained_models:
        y_points1, y_points2 = build_eval_points()
        cross_pred, cross_exact = print_cross_terms_summary(trained_models, y_points1, y_points2, device)
        
        # Escribir términos cruzados a CSV
        cross_csv_path = os.path.join(RESULTS_DIR_DATA, "cross_terms.csv")
        try:
            with open(cross_csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['term', 'predicted', 'exact', 'rel_err_pct', 'source_model'])
                for term in ['R12_eff', 'R13_eff', 'R23_eff']:
                    pred_val = cross_pred.get(term, float('nan'))
                    exact_val = cross_exact.get(term, float('nan'))
                    rel_err = abs(pred_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else float('nan')
                    source = 'p=2' if term == 'R12_eff' else 'p=3'
                    w.writerow([term, f"{pred_val:.6f}", f"{exact_val:.6f}", f"{rel_err:.4f}", source])
            print(f"{MAGENTA}{ICO_SAVE} Términos cruzados guardados: {cross_csv_path}{RESET}")
        except Exception as e:
            print(f"{YELLOW}Error guardando CSV términos cruzados: {e}{RESET}")
    elif 2 in trained_models or 3 in trained_models:
        print(f"\n{YELLOW}⚠️ Para calcular todos los términos cruzados se necesitan modelos p=2 y p=3{RESET}")
        # Mostrar lo que se pueda
        y_points1, y_points2 = build_eval_points()
        cross_pred = extract_cross_terms_from_models(trained_models, y_points1, y_points2, device)
        cross_exact = effective_exact_cross_terms()
        if cross_pred:
            print(f"\n{BOLD}Términos cruzados disponibles:{RESET}")
            for term, pred_val in cross_pred.items():
                exact_val = cross_exact[term]
                rel_err = abs(pred_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else float('nan')
                print(f"  {term}: pred={pred_val:.6f}, exacto={exact_val:.6f}, err={rel_err:.4f}%")
    
    # Escribir resumen extendido
    write_summary_csv_extended(results)
    
    # Mostrar resumen final
    print(f"\n{BOLD}{GREEN}{'='*90}{RESET}")
    print(f"{BOLD}{GREEN}RESUMEN FINAL{RESET}")
    print(f"{BOLD}{GREEN}{'='*90}{RESET}")
    print(f"Tiempo total: {format_elapsed(total_time)}")
    print(f"Tolerancia usada: {args.tolerance}%")
    print()
    
    header = f"{'Prob':<5} {'Ciclos':<8} {'Tiempo':<10} {'Pérdida':<14} {'Err.Rel %':<10} {'R_eff final':<14} {'R_eff exact':<14} {'Método':<10} {'Estado':<12}"
    print(header)
    print("-" * 100)
    for r in results:
        p = r['p']
        cycles = r.get('cycles', 0)
        converged = r.get('converged', False)
        method = r.get('reff_method', 'N/A')
        reff_final = r.get('reff_final', r['preds'][p-1] if 'preds' in r else float('nan'))
        reff_exact = r.get('reff_exact', r['exacts'][p-1] if 'exacts' in r else float('nan'))
        rel_err = r.get('rel_errs', [float('nan')]*6)[p-1] if not r.get('loaded') else abs(reff_final - reff_exact) / abs(reff_exact) * 100.0 if reff_exact != 0 else float('nan')
        
        if r.get('loaded'):
            status = "cargado"
            method = "loaded"
        elif converged:
            status = "convergido"
        else:
            status = "max ciclos"
        
        print(f"p={p:<3} {cycles:<8} {format_elapsed(r['time_sec']):<10} {r['final_loss']:<14.6e} "
              f"{rel_err:<10.4f} {reff_final:<14.6f} {reff_exact:<14.6f} {method:<10} {status:<12}")
    
    print(f"\n{GREEN}{ICO_OK} Entrenamiento completado. Resultados en: {RESULTS_DIR}{RESET}")


def write_summary_csv_extended(results: List[Dict[str, Any]]):
    """Escribe CSV resumen extendido con información de convergencia."""
    csv_path = os.path.join(RESULTS_DIR_DATA, "summary.csv")
    
    headers = ["problem", "cycles", "converged", "time_sec", "total_epochs", "final_loss",
               "reff_final", "reff_exact", "rel_err_pct", "reff_method"]
    
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            p = r['p']
            reff_final = r.get('reff_final', float('nan'))
            reff_exact = r.get('reff_exact', float('nan'))
            rel_err = abs(reff_final - reff_exact) / abs(reff_exact) * 100.0 if reff_exact != 0 else float('nan')
            
            row = [
                p,
                r.get('cycles', 0),
                r.get('converged', False),
                f"{r['time_sec']:.4f}",
                r.get('total_epochs', 0),
                f"{r['final_loss']:.6e}",
                f"{reff_final:.6f}",
                f"{reff_exact:.6f}",
                f"{rel_err:.4f}",
                r.get('reff_method', 'N/A')
            ]
            w.writerow(row)
    
    print(f"{MAGENTA}{ICO_SAVE} Resumen guardado: {csv_path}{RESET}")
    return csv_path


if __name__ == '__main__':
    main()
