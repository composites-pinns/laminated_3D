import torch
from .model import PINN
from .config import VF_mat1, VF_mat2


def load_models(device: torch.device):
    models = {}
    for p in range(1, 7):
        model = PINN(None, None, None, None, None, param={'VF_mat1': VF_mat1, 'VF_mat2': VF_mat2, 'R_mat1': None, 'R_mat2': None}, layers=[1, 50, 50, 3], device=device, prob_ind=p)
        # Note: We don't actually use this path in the refactor; the main script will instantiate properly.
        models[p] = model
    return models


def effective_properties(models: list[PINN], y_points1, y_points2, device: torch.device):
    # Compute effective properties using trained models
    R_eff = []
    for m in models:
        R_eff.append(m.Effective_Properties(y_points1, y_points2, device))
    return R_eff
