import torch
from .model import PINN
from .config import LAYERS, EPOCH_ADAMS, EPOCH_LBFGS, LOG_STEP
from .checkpoint import get_latest_checkpoint, load_into_model

# Colores ANSI e iconos (duplicados mínimos para no depender del main)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
GRAY = "\033[90m"
ICO_STEP = "🔧"
ICO_SKIP = "⏭️"
ICO_TL = "🔁"


def train_all_problems(
    y_collocation_mat1, 
    y_collocation_mat2, 
    y_interface, 
    y_uniqueness1, 
    y_uniqueness2, 
    param, 
    device: torch.device,
    training_order: list = None,
    existing_models: dict = None,
    use_transfer_learning: bool = True,
    retrain_targets: list | set | None = None,
    progress: bool = False
):
    """
    Train PINN models for elastic laminated composites.
    
    Args:
        training_order: List of problem indices to train (e.g., [6, 4, 5, 3, 2, 1])
                       If None, trains in default order [1, 2, 3, 4, 5, 6]
        existing_models: Dict of {prob_ind: model} for already-trained problems to skip
        use_transfer_learning: If True, initialize hardest problem with weights from easiest
    
    Returns:
        Dict of {prob_ind: model} with all trained models
    """
    if training_order is None:
        training_order = list(range(1, 7))
    
    if existing_models is None:
        existing_models = {}
    
    models = existing_models.copy()  # Start with existing models
    retrain_set = set(retrain_targets) if retrain_targets else set()

    # Log training order for clarity
    print(f"    {GRAY}Orden de entrenamiento{RESET}: {training_order}")

    N = len(training_order)
    for idx, p in enumerate(training_order, start=1):
        # Si el modelo ya existe en memoria en esta ejecución, no lo reentrenamos
        if p in models and p not in retrain_set:
            print(f"    {ICO_SKIP} [Paso {idx}/{N}] Omitiendo p={p} (ya entrenado y congelado)")
            continue

        # Si hay un checkpoint existente, cargarlo y evitar reentrenar salvo que esté en retrain_targets
        if p not in models and p not in retrain_set:
            latest = get_latest_checkpoint(p)
            if latest is not None:
                ckpt_path, ckpt_ver = latest
                print(f"    {ICO_SKIP} [Paso {idx}/{N}] Cargando checkpoint existente p={p} v{ckpt_ver}: {ckpt_path}")
                m = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
                         param=param, layers=LAYERS, device=device, prob_ind=p)
                m.to(device)
                load_into_model(m, ckpt_path)
                models[p] = m
                continue

        is_retrain = p in models and p in retrain_set
        action_str = "Reentrenando" if (is_retrain or p in retrain_set) else "Entrenando"
        print(f"    {BOLD}{BLUE}{ICO_STEP} [Paso {idx}/{N}] {action_str} problema p={p}{RESET}")

        m = PINN(y_collocation_mat1, y_collocation_mat2, y_interface, y_uniqueness1, y_uniqueness2,
                 param=param, layers=LAYERS, device=device, prob_ind=p)
        m.to(device)

        if is_retrain:
            # Inicializar desde el propio modelo existente para fine-tuning
            m.fcn_mat1.load_state_dict(models[p].fcn_mat1.state_dict())
            m.fcn_mat2.load_state_dict(models[p].fcn_mat2.state_dict())
        else:
            # Transfer learning: usar el primer modelo disponible como inicialización
            if use_transfer_learning and len(models) > 0:
                source_p = training_order[0] if training_order[0] in models else list(models.keys())[0]
                m.fcn_mat1.load_state_dict(models[source_p].fcn_mat1.state_dict())
                m.fcn_mat2.load_state_dict(models[source_p].fcn_mat2.state_dict())

        m.train(nIter_Adam=EPOCH_ADAMS, nIter_LBFGS=EPOCH_LBFGS, log_every=LOG_STEP, progress=progress)
        models[p] = m
    
    # Return as ordered list for compatibility
    # Devolver siempre una lista de 6 posiciones [p=1..6].
    # Si algún problema no se entrenó en esta llamada, la posición correspondiente será None
    # o el modelo existente si venía en existing_models.
    return [models.get(i) for i in range(1, 7)]
