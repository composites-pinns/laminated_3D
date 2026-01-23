#!/usr/bin/env python3
"""
Main orchestration script for training and evaluating PINNs for laminated
composites.

This script coordinates progressive training sequences and checkpointing for
laminated-composite PINNs. The following sections explain, in both English and
Spanish, how per-problem versioning works and how to re-train/refine models.

English
-------
- Per-problem versioning and refinement
    - Each subproblem `p` (1..6) is checkpointed with a version number using
        the filename pattern: ``model_laminated3d_pinns_{p}_v{version}.pt``. Files
        are stored in `CHECKPOINT_DIR`.
    - Versions represent successive refinements of the model for the same
        target property. For example, ``v1`` is typically the first completed
        training for `p`, and later sequences may produce ``v2``, ``v3`` etc. as
        the model is re-trained or fine-tuned using different transfer-learning
        subsequences.
    - Utility helpers in the `pinn.checkpoint` module are used to manage this
        scheme:
        - `get_latest_checkpoint(p, CHECKPOINT_DIR)` -> returns `(path, version)`
            for the most recent checkpoint of problem `p`, or `None` if none exist.
        - `get_next_version(p, CHECKPOINT_DIR)` and `checkpoint_path_for(p, v, dir)`
            compute the next version number and path prior to saving.
    - To re-train a single problem `p` from scratch, remove its checkpoint files
        (e.g. delete `model_laminated3d_pinns_{p}_v*.pt` from `CHECKPOINT_DIR`). On
        the next run the script will treat `p` as untrained and will create a new
        versioned checkpoint when finished.
    - The script also accepts `--force-retrain`, which purges recognized
        checkpoints at startup to force a full re-training run.
    - The training helper `train_all_problems` supports a `retrain_targets`
        argument (used in the script) to request retraining only for selected
        target problems while keeping other checkpoints frozen.

        
Español
-------
Resumen
------
Este script coordina secuencias progresivas de entrenamiento y el
guardado de checkpoints para PINNs de laminados. Implementa un esquema de
versionado por problema que permite refinar de forma independiente cada
propiedad objetivo mientras se mantienen congelados otros modelos para
transferencia de aprendizaje.

Checkpoints y versionado
------------------------
- Nombres de fichero: los checkpoints se escriben en `CHECKPOINT_DIR` usando
    el patrón ``model_laminated3d_pinns_{p}_v{version}.pt``, donde ``p`` es el
    índice del problema (1..6) y ``version`` es un contador entero.
- Significado de las versiones: las versiones representan refinamientos
    sucesivos de la misma propiedad objetivo. Por ejemplo, ``v1`` suele ser el
    primer entrenamiento completado para el problema ``p``; ejecuciones de
    afinado posteriores generan ``v2``, ``v3``, etc.
- Utilidades: el módulo `pinn.checkpoint` proporciona helpers usados aquí:
    - `get_latest_checkpoint(p, CHECKPOINT_DIR)` devuelve `(path, version)` del
        checkpoint más reciente para `p`, o `None` si no existe.
    - `get_next_version(p, CHECKPOINT_DIR)` y `checkpoint_path_for(p, v, dir)`
        calculan el próximo número de versión y la ruta destino antes de guardar.

Reentrenamiento y afinamientos
-------------------------------
- Para reentrenar un único problema `p` desde cero, elimine sus archivos de
    checkpoint (p. ej. borrar ficheros que coincidan con
    ``model_laminated3d_pinns_{p}_v*.pt`` en `CHECKPOINT_DIR`). El script
    tratará a `p` como no entrenado en la siguiente ejecución y guardará una
    nueva versión al finalizar.
- El script acepta `--force-retrain`, que purga los checkpoints reconocidos al
    iniciar y fuerza un reentrenamiento completo.
- La función de entrenamiento `train_all_problems` soporta el parámetro
    `retrain_targets` (empleado en este script) para reentrenar solo ciertos
    objetivos manteniendo otros modelos congelados para transferencia de
    aprendizaje.

Secuencias y configuración manual
--------------------------------
El orden de los subproblemas y las subsecuencias de transferencia se definen
manualmente dentro de este script (ver `sequence_map` y los argumentos
`training_order` pasados a `train_all_problems`). Actualmente las secuencias se
editan a mano; si desea experimentar fácilmente considere externalizarlas a un
archivo de configuración o exponerlas por la línea de comandos.

Ejecución determinista y reproducibilidad
-----------------------------------------
El script configura semillas aleatorias, ajustes deterministas de PyTorch y
algunas variables de entorno (ver el código más abajo) para minimizar la
variabilidad entre ejecuciones. Esto mejora la reproducibilidad, aunque puede
reducir el rendimiento en ciertos hardware.

Notas
-----
Esta cabecera documenta el comportamiento para que los usuarios puedan
controlar el reentrenamiento/versionado manipulando los ficheros de
checkpoint, usando `retrain_targets` o ejecutando con `--force-retrain`
cuando sea necesario.

English
-------
Overview
--------
This script coordinates progressive training sequences and checkpointing for
laminated-composite PINNs. It implements a per-problem versioning scheme so
that individual target properties can be refined repeatedly while keeping
other models frozen for transfer learning.

Checkpoints and versioning
--------------------------
- Filenames: checkpoints are written into `CHECKPOINT_DIR` using the pattern
    ``model_laminated3d_pinns_{p}_v{version}.pt`` where ``p`` is the problem
    index (1..6) and ``version`` is an integer version counter.
- Meaning of versions: versions represent successive refinements of the same
    target property. For example, ``v1`` is typically the first completed
    training for problem ``p``; later refinement runs produce ``v2``, ``v3``,
    etc.
- Utilities: the `pinn.checkpoint` module provides helpers used here:
    - `get_latest_checkpoint(p, CHECKPOINT_DIR)` returns `(path, version)` for
        the most recent checkpoint of `p`, or `None` if none exist.
    - `get_next_version(p, CHECKPOINT_DIR)` and `checkpoint_path_for(p, v, dir)`
        compute the next version number and the target path before saving.

Re-training and refinements
---------------------------
- To re-train a single problem `p` from scratch, delete its checkpoint files
    (e.g. remove files matching ``model_laminated3d_pinns_{p}_v*.pt`` from
    `CHECKPOINT_DIR`). The script will then treat `p` as untrained and will
    create a new versioned checkpoint on completion.
- The script accepts `--force-retrain`, which purges recognized checkpoints
    at startup and forces a full re-training run.
- The training helper `train_all_problems` supports a `retrain_targets`
    parameter (used in this script) to request retraining only for selected
    target problems while keeping other models frozen for transfer learning.

Sequences and manual configuration
----------------------------------
The ordering of subproblems and the transfer-learning subsequences are
defined manually within this script (see the `sequence_map` and the explicit
`training_order` arguments passed to `train_all_problems`). These sequences
are edited by hand; if you want easier experimentation consider externalizing
them to a configuration file or exposing them via CLI options.

Deterministic execution and reproducibility
-------------------------------------------
The script configures random seeds, PyTorch deterministic settings and a few
environment variables (see code below) to minimize run-to-run variability.
This improves reproducibility but may reduce performance on some hardware.

Notes
-----
This header documents behavior so users can control retraining/versioning by
manipulating checkpoint files, using `retrain_targets` or running with
`--force-retrain` when needed.
"""


import os
# Ensure reproducible cuBLAS behavior when using deterministic algorithms.
# This must be set before importing torch or initializing any CUDA context.
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import argparse
import random
import numpy as np
import torch
import time
import glob

from pinn.device import get_device
from pinn.config import TRAIN_SEED, VF_mat1, VF_mat2, DTYPE, CHECKPOINT_DIR
from pinn.materials import (
    R_mat1,
    R_mat2,
    R11_mat1, R12_mat1, R13_mat1, R22_mat1, R23_mat1, R33_mat1, R44_mat1, R55_mat1, R66_mat1,
    R11_mat2, R12_mat2, R13_mat2, R22_mat2, R23_mat2, R33_mat2, R44_mat2, R55_mat2, R66_mat2,
)
from pinn.data import build_train_data, build_eval_points
from pinn.train import train_all_problems
from pinn.checkpoint import get_latest_checkpoint, get_next_version, checkpoint_path_for, load_into_model

# Colores ANSI e iconos para logs en terminal
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"

ICO_SEQ = "🎯"
ICO_ORDER = "🧭"
ICO_FROZEN = "🧊"
ICO_DONE = "✅"
ICO_TITLE = "📚"
ICO_TABLE = "📊"
ICO_SKIP = "⏭️"
ICO_FORCE = "🔁"
ICO_TIME = "⏱️"


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento progresivo de PINNs laminados")
    parser.add_argument("--force-retrain", action="store_true", help="Ignorar checkpoints y reentrenar todas las secuencias")
    parser.add_argument("--only", type=int, choices=range(1,7), help="Ejecutar únicamente la secuencia cuyo objetivo final es el problema p (1-6)")
    args = parser.parse_args()

    # ========================================================================
    # CONFIGURACIÓN DETERMINISTA COMPLETA - ELIMINAR TODA ALEATORIEDAD
    # ========================================================================
    
    # 1. Establecer variables de entorno para determinismo antes de cualquier inicialización
    os.environ["PYTHONHASHSEED"] = str(TRAIN_SEED)
    # Note: CUBLAS_WORKSPACE_CONFIG is configured at module import time
    # (before torch is imported) to ensure cuBLAS deterministic behavior.

    # 2. Fijar seeds de todas las bibliotecas ANTES de usar device o crear tensores
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    
    # 3. Device setup (después de fijar seeds básicas)
    device = get_device()
    dev_str = str(device)
    if device.type == 'cuda':
        dev_icon = "🧠 CUDA"
        try:
            torch.cuda.manual_seed_all(TRAIN_SEED)  # Fijar seed CUDA si está disponible
        except Exception:
            pass
    else:
        dev_icon = "🖥️ CPU"
    print(f"{BOLD}{CYAN}Usando dispositivo{RESET}: {BOLD}{dev_icon}{RESET} ({dev_str})")

    # 4. Fijar dtype por defecto según config (float32 en MPS, float64 en otros)
    torch.set_default_dtype(DTYPE)

    # 5. Configurar algoritmos deterministas (siempre)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # 6. Ajustes de backend según modo
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

    # 7. Reducir variabilidad por paralelismo (opcional pero recomendado)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    
    print(f"{GREEN}✓ Configuración determinista activada (SEED={TRAIN_SEED}){RESET}")

    # Si se solicita reentrenamiento global, purgar checkpoints para empezar de cero
    if args.force_retrain:
        ckpt_pattern = os.path.join(CHECKPOINT_DIR, "model_laminated3d_pinns_*_v*.pt")
        paths = sorted(glob.glob(ckpt_pattern))
        if paths:
            print(f"{YELLOW}--force-retrain: eliminando {len(paths)} checkpoints en '{CHECKPOINT_DIR}'{RESET}")
            for p in paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
        else:
            print(f"{YELLOW}--force-retrain: no se encontraron checkpoints previos en '{CHECKPOINT_DIR}'{RESET}")

    # Params
    PARAM = {'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': R_mat1, 'R_mat2': R_mat2}

    # Data
    y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2 = build_train_data()
    y_points1, y_points2 = build_eval_points()

    # Progressive training strategy: train each problem with its optimal order
    # Each problem gets trained last in its sequence for best results
    print("\n" + CYAN + "="*60 + RESET)
    if args.only:
        print(f"{BOLD}{ICO_TITLE} MODO SOLO SECUENCIA p={args.only}{RESET}")
    else:
        print(f"{BOLD}{ICO_TITLE} ESTRATEGIA PROGRESIVA DE ENTRENAMIENTO{RESET}")
    print(CYAN + "="*60 + RESET)
    
    trained_models = {}  # Dict to store finalized models

    # Helper para guardar checkpoint al final de una secuencia
    def save_sequence_checkpoint(p_target: int, model):
        version = get_next_version(p_target, CHECKPOINT_DIR)
        path = checkpoint_path_for(p_target, version, CHECKPOINT_DIR)
        state = getattr(model, 'best_state', None)
        if state is None:
            state = model.state_dict()
        torch.save(state, path)
        print(f"    {GREEN}{ICO_DONE} Guardado checkpoint p={p_target} v{version}: {path}{RESET}")

    # Helper para cargar un checkpoint y crear el modelo
    def load_model_from_ckpt(p_target: int):
        latest = get_latest_checkpoint(p_target, CHECKPOINT_DIR)
        if latest is None:
            return None, None, None
        ckpt_path, ckpt_ver = latest
        from pinn.config import LAYERS
        from pinn.model import PINN
        m = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
                 param=PARAM, layers=LAYERS, device=device, prob_ind=p_target)
        m.to(device)
        load_into_model(m, ckpt_path)
        return m, ckpt_ver, ckpt_path

    # Helper para obtener latest checkpoint respetando --force-retrain
    def latest_ckpt_if_not_forced(p: int):
        existing = get_latest_checkpoint(p, CHECKPOINT_DIR)
        if args.force_retrain and existing is not None:
            path, ver = existing
            print(f"    {YELLOW}{ICO_FORCE} Forzando reentrenamiento: ignorando checkpoint p={p} v{ver} -> {path}{RESET}")
            return None
        return existing

    # Si se solicita sólo una secuencia, ejecutar flujo genérico y salir
    if args.only:
        seq_t0 = time.perf_counter()
        sequence_map = {
            6: [6],
            5: [6,5],
            4: [6,5,4],
            3: [6,4,5,3],
            2: [6,4,5,3,2],
            1: [6,4,5,3,2,1]
        }
        target = args.only
        order = sequence_map[target]
        print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia única: optimizar R{target}{RESET}")
        latest_target = latest_ckpt_if_not_forced(target)
        if latest_target is not None and not args.force_retrain:
            m, ver, path = load_model_from_ckpt(target)
            trained_models[target] = m
            print(f"    {ICO_SKIP} Saltando secuencia única (p={target} ya entrenado v{ver} -> {path})")
        else:
            # Preparar modelos existentes (congelados) para orden
            existing_models = {}
            frozen_list = []
            for q in order[:-1]:  # problemas anteriores en la secuencia
                if not args.force_retrain:
                    latest_q = get_latest_checkpoint(q, CHECKPOINT_DIR)
                    if latest_q is not None:
                        mq, vq, pathq = load_model_from_ckpt(q)
                        existing_models[q] = mq
                        frozen_list.append(f"p={q} v{vq}")
            if frozen_list:
                print(f"    {ICO_FROZEN} Modelos previos congelados: " + ", ".join(frozen_list))
            print(f"    {ICO_ORDER} Subsecuencia: " + " → ".join(str(x) for x in order))
            models_all = train_all_problems(
                y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
                PARAM, device,
                training_order=order,
                existing_models=existing_models.copy(),
                use_transfer_learning=True,
                progress=False
            )
            trained_models[target] = models_all[target-1]
            save_sequence_checkpoint(target, trained_models[target])
            print(f"    {GREEN}{ICO_DONE} p={target} entrenado y {ICO_FROZEN} congelado{RESET}")
        # Sólo evaluamos el coeficiente objetivo si no tenemos todos los modelos
        if len(trained_models) == 6:
            pass  # caerá al bloque general de evaluación si existiera, pero vamos a terminar
        else:
            # Evaluación específica del coeficiente objetivo para diagnóstico rápido entre dispositivos
            from pinn.data import build_eval_points as _build_eval_points
            y_points1_, y_points2_ = _build_eval_points()
            model_target = trained_models.get(target)
            if model_target is not None:
                eff_tuple = model_target.Effective_Properties(y_points1_, y_points2_, device)
                # eff_tuple = (R1p, R2p, ..., R6p) para prob_ind = target.
                # Queremos R_{target,target} = índice i=target en esta tupla (1-based)
                pred_eff = eff_tuple[target - 1]
                # Propiedad efectiva exacta para cada p
                def prom(CC1, CC2):
                    return VF_mat1 * CC1 + VF_mat2 * CC2
                # Cálculo exacto según fórmulas usadas en bloque general (simplificamos sólo la parte necesaria)
                if target == 6:
                    exact_eff = prom(R66_mat1, R66_mat2)
                    coef_name = 'R66'
                elif target == 5:
                    exact_eff = prom(1 / R55_mat1, 1 / R55_mat2) ** (-1)
                    coef_name = 'R55'
                elif target == 4:
                    exact_eff = prom(1 / R44_mat1, 1 / R44_mat2) ** (-1)
                    coef_name = 'R44'
                elif target == 3:
                    exact_eff = prom(1 / R33_mat1, 1 / R33_mat2) ** (-1)
                    coef_name = 'R33'
                elif target == 2:
                    # Fórmulas reducidas para h22x3 (ver bloque general)
                    exact_eff = prom(R22_mat1, R22_mat2) - prom(R23_mat1**2 / R33_mat1, R23_mat2**2 / R33_mat2) + prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2) ** 2 * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1)
                    coef_name = 'R22'
                else:  # target == 1
                    exact_eff = prom(R11_mat1, R11_mat2) - prom(R13_mat1**2 / R33_mat1, R13_mat2**2 / R33_mat2) + prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2) ** 2 / prom(1 / R33_mat1, 1 / R33_mat2)
                    coef_name = 'R11'
                rel_err = abs(pred_eff - exact_eff) / abs(exact_eff) * 100.0 if exact_eff != 0 else float('nan')
                print(f"{BOLD}{ICO_TABLE} Resultado {coef_name} (modo --only){RESET}")
                print(f"Predicho: {pred_eff:.6f} | Exacto: {exact_eff:.6f} | RelErr (%): {rel_err:.4f}")
            print(f"{YELLOW}Finalizado modo --only. Para tabla completa ejecute sin --only.{RESET}")
        seq_dt = time.perf_counter() - seq_t0
        def _fmt_dur(s):
            s = float(s)
            m, s = divmod(int(round(s)), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        print(f"    {BLUE}{ICO_TIME} Tiempo secuencia (p={target}): {_fmt_dur(seq_dt)}{RESET}")
        return

    # Secuencia 1: optimizar R66 (p=6) - primero
    print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 1: optimizar R66 (p=6){RESET}")
    seq1_t0 = time.perf_counter()
    latest = latest_ckpt_if_not_forced(6)
    if latest is not None:
        m, ver, path = load_model_from_ckpt(6)
        trained_models[6] = m
        print(f"    {ICO_SKIP} Saltando secuencia 1 (p=6 ya entrenado v{ver} -> {path})")
    else:
        print(f"    {ICO_ORDER} Subsecuencia: 6")
        models = train_all_problems(
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
            PARAM, device,
            training_order=[6],
            existing_models={},
            use_transfer_learning=False,
            progress=False
        )
        trained_models[6] = models[5]
        save_sequence_checkpoint(6, trained_models[6])
        print(f"    {GREEN}{ICO_DONE} p=6 (R66) entrenado y {ICO_FROZEN} congelado{RESET}")
    seq1_dt = time.perf_counter() - seq1_t0
    def _fmt_dur(s):
        s = float(s)
        m, s = divmod(int(round(s)), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 1: {_fmt_dur(seq1_dt)}{RESET}")

    # Secuencia 2: optimizar R55 (p=5)
    print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 2: optimizar R55 (p=5){RESET}")
    seq2_t0 = time.perf_counter()
    latest = latest_ckpt_if_not_forced(5)
    if latest is not None:
        m, ver, path = load_model_from_ckpt(5)
        trained_models[5] = m
        print(f"    {ICO_SKIP} Saltando secuencia 2 (p=5 ya entrenado v{ver} -> {path})")
    else:
        print(f"    {ICO_FROZEN} Modelos previos congelados: p=6")
        print(f"    {ICO_ORDER} Subsecuencia: 6 → 5")
        models = train_all_problems(
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
            PARAM, device,
            training_order=[6, 5],
            existing_models=trained_models.copy(),
            use_transfer_learning=True,
            progress=False
        )
        trained_models[5] = models[4]
        save_sequence_checkpoint(5, trained_models[5])
        print(f"    {GREEN}{ICO_DONE} p=5 (R55) entrenado y {ICO_FROZEN} congelado{RESET}")
    seq2_dt = time.perf_counter() - seq2_t0
    print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 2: {_fmt_dur(seq2_dt)}{RESET}")

    # Secuencia 3: optimizar R44 (p=4)
    print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 3: optimizar R44 (p=4){RESET}")
    seq3_t0 = time.perf_counter()
    latest = latest_ckpt_if_not_forced(4)
    if latest is not None:
        m, ver, path = load_model_from_ckpt(4)
        trained_models[4] = m
        print(f"    {ICO_SKIP} Saltando secuencia 3 (p=4 ya entrenado v{ver} -> {path})")
    else:
        print(f"    {ICO_FROZEN} Modelos previos congelados: p=6, p=5")
        print(f"    {ICO_ORDER} Subsecuencia: 6 → 5 → 4")
        models = train_all_problems(
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
            PARAM, device,
            training_order=[6, 5, 4],
            existing_models=trained_models.copy(),
            use_transfer_learning=True,
            progress=False
        )
        trained_models[4] = models[3]
        save_sequence_checkpoint(4, trained_models[4])
        print(f"    {GREEN}{ICO_DONE} p=4 (R44) entrenado y {ICO_FROZEN} congelado{RESET}")
    seq3_dt = time.perf_counter() - seq3_t0
    print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 3: {_fmt_dur(seq3_dt)}{RESET}")

    # Secuencia 4: optimizar R33 (p=3)
    print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 4: optimizar R33 (p=3){RESET}")
    seq4_t0 = time.perf_counter()
    latest = latest_ckpt_if_not_forced(3)
    if latest is not None:
        m, ver, path = load_model_from_ckpt(3)
        trained_models[3] = m
        print(f"    {ICO_SKIP} Saltando secuencia 4 (p=3 ya entrenado v{ver} -> {path})")
    else:
        print(f"    {ICO_FROZEN} Modelos previos congelados: p=6, p=5, p=4")
        print(f"    {ICO_ORDER} Subsecuencia: 6 → 4 → 5 → 3")
        models = train_all_problems(
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
            PARAM, device,
            training_order=[6, 4, 5, 3],
            existing_models=trained_models.copy(),
            use_transfer_learning=True,
            progress=False
        )
        trained_models[3] = models[2]
        save_sequence_checkpoint(3, trained_models[3])
        print(f"    {GREEN}{ICO_DONE} p=3 (R33) entrenado y {ICO_FROZEN} congelado{RESET}")
    seq4_dt = time.perf_counter() - seq4_t0
    print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 4: {_fmt_dur(seq4_dt)}{RESET}")

    # Secuencia 5: optimizar R22 (p=2)
    print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 5: optimizar R22 (p=2){RESET}")
    seq5_t0 = time.perf_counter()
    latest = latest_ckpt_if_not_forced(2)
    if latest is not None:
        m, ver, path = load_model_from_ckpt(2)
        trained_models[2] = m
        print(f"    {ICO_SKIP} Saltando secuencia 5 (p=2 ya entrenado v{ver} -> {path})")
    else:
        print(f"    {ICO_FROZEN} Modelos previos congelados: p=6, p=5, p=4, p=3")
        print(f"    {ICO_ORDER} Subsecuencia: 6 → 4 → 5 → 3 → 2")
        models = train_all_problems(
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
            PARAM, device,
            training_order=[6, 4, 5, 3, 2],
            existing_models=trained_models.copy(),
            use_transfer_learning=True,
            progress=False
        )
        trained_models[2] = models[1]
        save_sequence_checkpoint(2, trained_models[2])
        print(f"    {GREEN}{ICO_DONE} p=2 (R22) entrenado y {ICO_FROZEN} congelado{RESET}")
    seq5_dt = time.perf_counter() - seq5_t0
    print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 5: {_fmt_dur(seq5_dt)}{RESET}")

    # Secuencia 6: optimizar R11 (p=1)
    print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 6: optimizar R11 (p=1){RESET}")
    seq6_t0 = time.perf_counter()
    latest = latest_ckpt_if_not_forced(1)
    if latest is not None:
        m, ver, path = load_model_from_ckpt(1)
        trained_models[1] = m
        print(f"    {ICO_SKIP} Saltando secuencia 6 (p=1 ya entrenado v{ver} -> {path})")
    else:
        print(f"    {ICO_FROZEN} Modelos previos congelados: p=6, p=5, p=4, p=3, p=2")
        print(f"    {ICO_ORDER} Subsecuencia: 6 → 4 → 5 → 3 → 2 → 1")
        models = train_all_problems(
            y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
            PARAM, device,
            training_order=[6, 4, 5, 3, 2, 1],
            existing_models=trained_models.copy(),
            use_transfer_learning=True,
            progress=False
        )
        trained_models[1] = models[0]
        save_sequence_checkpoint(1, trained_models[1])
        print(f"    {GREEN}{ICO_DONE} p=1 (R11) entrenado y {ICO_FROZEN} congelado{RESET}")
    seq6_dt = time.perf_counter() - seq6_t0
    print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 6: {_fmt_dur(seq6_dt)}{RESET}")


    # # Secuencia 7: Afinar R_55 (p=5)
    # print(f"\n{BOLD}{MAGENTA}{ICO_SEQ} Secuencia 7: afinar R55 (p=5){RESET}")
    # seq7_t0 = time.perf_counter()
    # # Evitar ejecutar esta secuencia antes de que p=4 esté entrenado
    # if (4 not in trained_models) and (not args.force_retrain):
    #     print(f"    {ICO_SKIP} Saltando secuencia 7 (p=4 aún no entrenado; se ejecutará al final)")
    #     seq7_dt = 0.0
    #     print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 7: {_fmt_dur(seq7_dt)}{RESET}")
    # else:
    #     # Si ya existe p=5 v2 y no estamos forzando, saltar
    #     v2_path = checkpoint_path_for(5, 2, CHECKPOINT_DIR)
    #     if (not args.force_retrain) and os.path.exists(v2_path):
    #         print(f"    {ICO_SKIP} Saltando secuencia 7 (p=5 v2 ya existe -> {v2_path})")
    #         # Cargar p=5 v2 para evaluación final
    #         from pinn.config import LAYERS
    #         from pinn.model import PINN
    #         m = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
    #                  param=PARAM, layers=LAYERS, device=device, prob_ind=5)
    #         m.to(device)
    #         load_into_model(m, v2_path)
    #         trained_models[5] = m
    #         seq7_dt = time.perf_counter() - seq7_t0
    #         print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 7: {_fmt_dur(seq7_dt)}{RESET}")
    #     else:
    #         # Reentrenar p=5 para crear v2 usando TL desde p=6/p=4 (excluimos p=5 actual)
    #         existing_models = trained_models.copy()
    #         if 5 in existing_models:
    #             existing_models.pop(5)
    #         frozen_list = [f"p={k}" for k in sorted(existing_models.keys())]
    #         if frozen_list:
    #             print(f"    {ICO_FROZEN} Modelos previos congelados: " + ", ".join(frozen_list))
    #         print(f"    {ICO_ORDER} Subsecuencia: 6 → 4 → 5")
    #         models = train_all_problems(
    #             y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
    #             PARAM, device,
    #             training_order=[6, 4, 5],
    #             existing_models=existing_models,
    #             use_transfer_learning=True,
    #             retrain_targets=[5],
    #             progress=False
    #         )
    #         trained_models[5] = models[2]
    #         # Guardar exactamente como versión 2
    #         version = 2
    #         path = checkpoint_path_for(5, version, CHECKPOINT_DIR)
    #         state = getattr(trained_models[5], 'best_state', None)
    #         if state is None:
    #             state = trained_models[5].state_dict()
    #         torch.save(state, path)
    #         print(f"    {GREEN}{ICO_DONE} p=5 (R55 afinado) entrenado y {ICO_FROZEN} congelado{RESET}")
    #         print(f"    {GREEN}{ICO_DONE} Guardado checkpoint p=5 v{version}: {path}{RESET}")
    #         seq7_dt = time.perf_counter() - seq7_t0
    #         print(f"    {BLUE}{ICO_TIME} Tiempo secuencia 7: {_fmt_dur(seq7_dt)}{RESET}")
    #     # (Se eliminaron líneas espurias que entrenaban p=3 aquí por error)

    # Reconstruct final models list in order
    models = [trained_models[i] for i in range(1, 7)]

    print("\n" + CYAN + "="*60 + RESET)
    print(f"{BOLD}{GREEN}TODOS LOS MODELOS ENTRENADOS EN SUS RESPECTIVAS SECUENCIAS {RESET}")
    print(CYAN + "="*60 + RESET + "\n")

    # Tiempo total
    total_dt = seq1_dt + seq2_dt + seq3_dt + seq4_dt + seq5_dt + seq6_dt #+ seq7_dt
    print(f"\n{BLUE}{ICO_TIME} Tiempo total entrenamiento: {_fmt_dur(total_dt)}{RESET}")

    # Evaluate effective properties
    def prom(CC1, CC2):
        return VF_mat1 * CC1 + VF_mat2 * CC2

    # Compute predicted effective properties using each model
    preds = [m.Effective_Properties(y_points1, y_points2, device) for m in models]
    (
        (R11_eff_pred, R21_eff_pred, R31_eff_pred, R41_eff_pred, R51_eff_pred, R61_eff_pred),
        (R12_eff_pred, R22_eff_pred, R32_eff_pred, R42_eff_pred, R52_eff_pred, R62_eff_pred),
        (R13_eff_pred, R23_eff_pred, R33_eff_pred, R43_eff_pred, R53_eff_pred, R63_eff_pred),
        (R14_eff_pred, R24_eff_pred, R34_eff_pred, R44_eff_pred, R54_eff_pred, R64_eff_pred),
        (R15_eff_pred, R25_eff_pred, R35_eff_pred, R45_eff_pred, R55_eff_pred, R65_eff_pred),
        (R16_eff_pred, R26_eff_pred, R36_eff_pred, R46_eff_pred, R56_eff_pred, R66_eff_pred),
    ) = preds

    # Exact effective properties (Castillero IJSS-1998)
    h11x3 = prom(R11_mat1, R11_mat2) - prom(R13_mat1**2 / R33_mat1, R13_mat2**2 / R33_mat2) + prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2) ** 2 / prom(1 / R33_mat1, 1 / R33_mat2)
    h12x3 = prom(R12_mat1, R12_mat2) - prom(R13_mat1 * R23_mat1 / R33_mat1, R13_mat2 * R23_mat2 / R33_mat2) + prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2) * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1) * prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2)
    h13x3 = prom(R13_mat1 / R33_mat1, R13_mat2 / R33_mat2) * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1)
    h22x3 = prom(R22_mat1, R22_mat2) - prom(R23_mat1**2 / R33_mat1, R23_mat2**2 / R33_mat2) + prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2) ** 2 * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1)
    h23x3 = prom(R23_mat1 / R33_mat1, R23_mat2 / R33_mat2) * (prom(1 / R33_mat1, 1 / R33_mat2)) ** (-1)
    h33x3 = prom(1 / R33_mat1, 1 / R33_mat2) ** (-1)
    h44x3 = prom(1 / R44_mat1, 1 / R44_mat2) ** (-1)
    h55x3 = prom(1 / R55_mat1, 1 / R55_mat2) ** (-1)
    h66x3 = prom(R66_mat1, R66_mat2)

    # Tabla alineada: Coef | Predicted | Exact | RelErr (%)
    def rel_err(p, e):
        return float(abs(p - e) / abs(e) * 100.0) if float(e) != 0.0 else float('nan')

    rows = [
        ("R11", float(R11_eff_pred), float(h11x3)),
        ("R12", float(R12_eff_pred), float(h12x3)),
        ("R13", float(R13_eff_pred), float(h13x3)),
        ("R22", float(R22_eff_pred), float(h22x3)),
        ("R23", float(R23_eff_pred), float(h23x3)),
        ("R33", float(R33_eff_pred), float(h33x3)),
        ("R44", float(R44_eff_pred), float(h44x3)),
        ("R55", float(R55_eff_pred), float(h55x3)),
        ("R66", float(R66_eff_pred), float(h66x3)),
    ]

    header = f"{'Coef':<6}{'Predicted':>14}{'Exact':>14}{'RelErr (%)':>14}"
    print(f"\n{BOLD}{ICO_TABLE} Resultados efectivos{RESET}")
    print(GRAY + header + RESET)
    print("-" * len(header))
    for name, pred, exact in rows:
        err = rel_err(pred, exact)
        # Coloreo por fila: verde si err < 0.1%, rojo si err > 1%
        row_color = GREEN if (not np.isnan(err) and err < 0.1) else (RED if (not np.isnan(err) and err > 1.0) else "")
        line = f"{name:<6}{pred:>14.6f}{exact:>14.6f}{err:>14.3f}"
        if row_color:
            print(f"{row_color}{line}{RESET}")
        else:
            print(line)


if __name__ == "__main__":
    main()

