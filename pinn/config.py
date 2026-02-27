import torch
from .device import get_device

TRAIN_SEED = 1234

DEVICE = get_device()

if DEVICE.type == 'mps':
    print("WARNING: Apple MPS does not fully support float64. Falling back to CPU with float64 precision.")
    DEVICE = torch.device('cpu')

DTYPE = torch.float64
torch.set_default_dtype(torch.float64)

LAYERS = [1, 50, 50, 3]

EPOCH_ADAMS = 2_000
EPOCH_LBFGS = 500

EPOCH_ADAMS_FOLLOWING_CYCLES = 100
EPOCH_LBFGS_FOLLOWING_CYCLES = 100

ADAM_PRINT_EVERY = EPOCH_ADAMS // 5
LBFGS_PRINT_EVERY = EPOCH_LBFGS // 5
ADAM_LATERCYCLES_PRINT_EVERY = EPOCH_ADAMS_FOLLOWING_CYCLES // 5
LBFGS_LATERCYCLES_PRINT_EVERY = EPOCH_LBFGS_FOLLOWING_CYCLES // 5

y_I = 0.2

VF_mat1 = y_I
VF_mat2 = 1 - VF_mat1

E1 = 206.74
E2 = 72.04

Gamma1 = 0.3
Gamma2 = 0.35

N_COLLOCATION = 400
N_UNIQUENESS = 2000
N_POINTS = 2000

NORMALIZE_INPUTS = True
ADAPTIVE_WEIGHTS = True
ADAPTIVE_BETA = 0.99
WEIGHT_MIN = 1e-6
WEIGHT_MAX = 1e6
WEIGHT_UNIQUENESS_MIN = 1.0

ADAM_LR = 1e-4

LBFGS_LR = 1.0
LBFGS_MAX_ITER = 10
LBFGS_MAX_EVAL_MULT = 1
LBFGS_HISTORY_SIZE = 10
LBFGS_LINE_SEARCH_FN = 'strong_wolfe'
LBFGS_TOL_GRAD = 1e-9
LBFGS_TOL_CHANGE = 1e-12

# NysNewton-CG parameters
EPOCH_NNCG = 50                    # iterations per cycle (first cycle)
EPOCH_NNCG_FOLLOWING_CYCLES = 50   # iterations per cycle (subsequent cycles)
NNCG_RANK = 10                     # rank of the Nyström sketch
NNCG_MU = 1e-4                     # Tikhonov regularisation
NNCG_CG_TOL = 1e-5                # CG relative residual tolerance
NNCG_MAX_CG_ITERS = 50            # max CG iterations per step
NNCG_LINE_SEARCH = True           # Armijo back-tracking line search
NNCG_LR = 1.0                     # learning rate / step size
NNCG_PRINT_EVERY = 10             # print cadence
NNCG_LATERCYCLES_PRINT_EVERY = 10 # print cadence for later cycles

# Convergence control
CONVERGENCE_WINDOW = 3             # moving-average window (number of cycles)
MIN_CYCLES_BEFORE_CHECK = 5       # first cycle at which convergence is evaluated
CV_WINDOW = 3                      # window for coefficient-of-variation criterion
