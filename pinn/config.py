# Global configuration and hyperparameters
import os
import torch
from .device import get_device

TRAIN_SEED = 1234
TEST_SEED = 5678

# Device and default dtype
# Enforce float64 everywhere. If the chosen device cannot support float64
# (notably Apple MPS), abort immediately with a clear error message.
DEVICE = get_device()

import sys

if DEVICE.type == 'mps':
    # MPS (Apple) does not support float64. Prefer CPU with float64 to
    # preserve numerical behavior required by this project.
    print("WARNING: Apple MPS does not fully support float64. Falling back to CPU with float64 precision.")
    DEVICE = torch.device('cpu')

# Force 64-bit precision globally
DTYPE = torch.float64
# Ensure tensors created without an explicit dtype default to float64
torch.set_default_dtype(torch.float64)

LAYERS = [1, 50, 50, 3]

# EPOCH_ADAMS = 1000  # baseline
EPOCH_ADAMS = 20000
# EPOCH_LBFGS = 10  # baseline
EPOCH_LBFGS = 500  # increased for finer refinement

LOG_STEP = 5

# Problem parameters

y_I = 0.2

# Volume fractions
VF_mat1 = y_I
VF_mat2 = 1 - VF_mat1

# Young modulus
E1 = 206.74
E2 = 72.04

# Poisson ratio
Gamma1 = 0.3
Gamma2 = 0.35

# Dataset sizes (tuneable)
# N_COLLOCATION = 200  # baseline: modest to avoid heavy runtimes
N_COLLOCATION = 400  # increased for better PDE enforcement
N_UNIQUENESS = 4000  #more points helps integral-based constraints
N_BOUNDARY = 1
N_POINTS = 10000  # baseline
#N_POINTS = 15000  # finer evaluation grid for more accurate integral of effective properties

# Normalization and adaptive loss weighting
NORMALIZE_INPUTS = True  # scale y to [-1, 1] inside the network
ADAPTIVE_WEIGHTS = True  # enable adaptive weighting across loss terms
# ADAPTIVE_BETA = 0.99  # baseline moving average smoothing
ADAPTIVE_BETA = 0.995  # smoother EMA for more stable weight updates
# WEIGHT_MIN = 0.2  # baseline
WEIGHT_MIN = 0.1  # allow more dynamic range
# WEIGHT_MAX = 5.0  # baseline
WEIGHT_MAX = 10.0  # allow stronger reweighting
WEIGHT_UPDATE_EVERY = 1  # iterations (during Adam phase only)

# Optimization
# ADAM_LR = 1e-3  # baseline
ADAM_LR = 5e-5  # lower LR for smoother convergence
LOG_WEIGHTS = False  # set True to log adaptive weights in training output

# Checkpoints directory
CHECKPOINT_DIR = 'checkpoints'

# Performance/fast-path flags have been removed. Precision is globally fixed to float64.

# Default sweep configuration for the grid/orchestrator. Can be customized by the user.
SWEEP_CONFIG = {
    # 10 sensible values for each sweep dimension (user-editable)
    # 'epochs_adams': [200, 400, 600, 800, 1000, 2000, 5000], #, 10000, 20000, 30000],
    # allow zero so the user can skip LBFGS in some experiments
    #'epochs_lbfgs': [0, 5, 10, 20, 30, 50, 80, 100, 200, 500, 1000],
    # log-spaced candidate learning rates centered around the default
    #'adam_lr': [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6],
    # a variety of small->large architectures (strings used by hyper_grid)
    #'layers': [
    #    '1-10-10-3', '1-20-20-3', '1-30-30-30-3', '1-50-50-3', '1-64-64-3',
    #    '1-100-50-3', '1-100-100-3', '1-200-200-3', '1-50-50-50-3', '1-300-300-3', '1-400-400-3', '1-500-500-3'
    #],
    # collocation grid sizes (increase by ~2x each step)
    'n_collocation': [400, 800, 1600, 3200],
    'targets': list(range(1, 7)),
}
