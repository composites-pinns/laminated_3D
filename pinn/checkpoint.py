import os
import re
import glob
import torch
from .config import CHECKPOINT_DIR


BASE_PATTERN = "model_laminated3d_pinns_{p}_v{v}.pt"
REGEX = re.compile(r"model_laminated3d_pinns_(?P<p>\d+)_v(?P<v>\d+)\.pt$")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_checkpoints_for(p: int, directory: str = CHECKPOINT_DIR):
    ensure_dir(directory)
    pattern = os.path.join(directory, BASE_PATTERN.format(p=p, v='*'))
    return sorted(glob.glob(pattern))


def get_latest_checkpoint(p: int, directory: str = CHECKPOINT_DIR):
    files = list_checkpoints_for(p, directory)
    if not files:
        return None
    # Parse versions and pick highest
    def ver(file):
        m = REGEX.search(os.path.basename(file))
        return int(m.group('v')) if m else -1
    files = sorted(files, key=ver)
    latest = files[-1]
    m = REGEX.search(os.path.basename(latest))
    v = int(m.group('v')) if m else 1
    return latest, v


def get_next_version(p: int, directory: str = CHECKPOINT_DIR) -> int:
    latest = get_latest_checkpoint(p, directory)
    if latest is None:
        return 1
    _, v = latest
    return v + 1


def checkpoint_path_for(p: int, version: int, directory: str = CHECKPOINT_DIR) -> str:
    ensure_dir(directory)
    return os.path.join(directory, BASE_PATTERN.format(p=p, v=version))


def load_into_model(model, path: str):
    state = torch.load(path, map_location=model.device, weights_only=True)
    model.load_state_dict(state)
    return model
