from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import time
from .config import (
    DTYPE, NORMALIZE_INPUTS, ADAPTIVE_WEIGHTS, ADAPTIVE_BETA,
    WEIGHT_MIN, WEIGHT_MAX, ADAM_LR,
    ADAM_PRINT_EVERY, LBFGS_PRINT_EVERY,
    LBFGS_LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL_MULT, LBFGS_HISTORY_SIZE,
    LBFGS_LINE_SEARCH_FN, LBFGS_TOL_GRAD, LBFGS_TOL_CHANGE,
    WEIGHT_UNIQUENESS_MIN,
    NNCG_RANK, NNCG_MU, NNCG_CG_TOL, NNCG_MAX_CG_ITERS,
    NNCG_LINE_SEARCH, NNCG_LR, NNCG_PRINT_EVERY,
)
from optimizer.nys_newton_cg import NysNewtonCG


class FCN(nn.Module):
    def __init__(self, layers: List[int], device: torch.device, activation: nn.Module | None = None):
        super().__init__()
        self.device = device
        self.layers = layers
        self.activation = activation or nn.Tanh()

        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1])
                                      for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        a = X.to(dtype=DTYPE)

        if NORMALIZE_INPUTS:
            a = 2.0 * a - 1.0

        for i in range(len(self.linear) - 1):
            z = self.linear[i](a)
            a = self.activation(z)
        return self.linear[-1](a)


class PINN(nn.Module):
    def __init__(
        self,
        y_collocation_mat1: np.ndarray,
        y_collocation_mat2: np.ndarray,
        y_interface: np.ndarray,
        y_uniqueness1: np.ndarray,
        y_uniqueness2: np.ndarray,
        param: dict,
        layers: List[int],
        device: torch.device,
        prob_ind: int = 1,
    ):
        super().__init__()
        self.device = device
        self.prob_ind = prob_ind

        # Collocation points
        self.y_mat1 = torch.tensor(y_collocation_mat1, requires_grad=True, dtype=DTYPE).to(self.device)
        self.y_mat2 = torch.tensor(y_collocation_mat2, requires_grad=True, dtype=DTYPE).to(self.device)
        self.null_mat1 = torch.zeros(self.y_mat1.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        self.null_mat2 = torch.zeros(self.y_mat2.shape, dtype=DTYPE, requires_grad=False).to(self.device)

        # Interface points
        self.y_inter = torch.tensor(y_interface, requires_grad=True, dtype=DTYPE).to(self.device)
        self.null_inter = torch.zeros(self.y_inter.shape, dtype=DTYPE, requires_grad=False).to(self.device)

        # Uniqueness condition
        self.y_uniq1 = torch.tensor(y_uniqueness1, requires_grad=False, dtype=DTYPE).to(self.device)
        self.y_uniq2 = torch.tensor(y_uniqueness2, requires_grad=False, dtype=DTYPE).to(self.device)

        # PDE parameters
        self.VF_mat1 = param['VF_mat1']
        self.VF_mat2 = param['VF_mat2']
        self.R_mat1 = param['R_mat1']
        self.R_mat2 = param['R_mat2']

        # NN
        self.layers = layers
        self.best = np.inf
        self.best_state = None  # Almacena el mejor estado (state_dict) para guardado al final de secuencia
        self.activation = nn.Tanh()

        self.fcn_mat1 = FCN(self.layers, self.device, activation=self.activation)
        self.fcn_mat2 = FCN(self.layers, self.device, activation=self.activation)

        self.fcn_mat1.to(self.device, dtype=DTYPE)
        self.fcn_mat2.to(self.device, dtype=DTYPE)

        # Optimizers
        self.params = list(self.fcn_mat1.parameters()) + list(self.fcn_mat2.parameters())
        self.optimizer_Adam = torch.optim.Adam(self.params, lr=ADAM_LR)

        self.max_iter_LBFGS = int(LBFGS_MAX_ITER)
        self._lbfgs_kwargs = {
            'lr': float(LBFGS_LR),
            'max_iter': self.max_iter_LBFGS,
            'max_eval': int(LBFGS_MAX_EVAL_MULT) * self.max_iter_LBFGS,
            'tolerance_grad': float(LBFGS_TOL_GRAD),
            'tolerance_change': float(LBFGS_TOL_CHANGE),
            'history_size': int(LBFGS_HISTORY_SIZE),
            'line_search_fn': (LBFGS_LINE_SEARCH_FN if LBFGS_LINE_SEARCH_FN in (None, 'strong_wolfe') else 'strong_wolfe'),
        }
        self.optimizer_LBFGS = torch.optim.LBFGS(self.params, **self._lbfgs_kwargs)

        # NysNewton-CG optimizer
        self._nncg_kwargs = {
            'lr': float(NNCG_LR),
            'rank': int(NNCG_RANK),
            'mu': float(NNCG_MU),
            'cg_tol': float(NNCG_CG_TOL),
            'max_cg_iters': int(NNCG_MAX_CG_ITERS),
            'line_search': bool(NNCG_LINE_SEARCH),
        }
        self.optimizer_NNCG = NysNewtonCG(self.params, **self._nncg_kwargs)

        # Losses
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.loss_log = []
        self.loss_collocation_log = []
        self.loss_interface_log = []
        self.loss_uniqueness_condition_log = []
        # Logs de contribuciones ponderadas (w * L) para verificar balance
        self.weighted_collocation_log = []
        self.weighted_uniqueness_log = []
        self.weighted_interface_log = []
        self.it = 0
        self.best_iter = 0
        self.log_every = 1
        self.pbar = None
        # Adaptive loss weights (initialized to 1)
        self.w_collocation = torch.tensor(1.0, dtype=DTYPE, device=self.device)
        self.w_uniqueness = torch.tensor(1.0, dtype=DTYPE, device=self.device)
        self.w_interface = torch.tensor(1.0, dtype=DTYPE, device=self.device)
        # Moving averages of each term
        self.mae_collocation = None
        self.mae_uniqueness = None
        self.mae_interface = None

    def reset_lbfgs(self):
        """Recreate LBFGS optimizer to reset its internal history/state."""
        self.optimizer_LBFGS = torch.optim.LBFGS(self.params, **self._lbfgs_kwargs)

    def reset_adam(self):
        """Recreate Adam optimizer to reset momentum/state."""
        self.optimizer_Adam = torch.optim.Adam(self.params, lr=ADAM_LR)

    def reset_nncg(self):
        """Recreate NysNewton-CG optimizer to reset its internal state."""
        self.optimizer_NNCG = NysNewtonCG(self.params, **self._nncg_kwargs)

    def reset_optimizers(self, adam: bool = True, lbfgs: bool = True, nncg: bool = True):
        if adam:
            self.reset_adam()
        if lbfgs:
            self.reset_lbfgs()
        if nncg:
            self.reset_nncg()

    def forward_mat1(self, points: torch.Tensor) -> torch.Tensor:
        return self.fcn_mat1(points)

    def forward_mat2(self, points: torch.Tensor) -> torch.Tensor:
        return self.fcn_mat2(points)

    def Loss_Uniqueness_Condition(self) -> torch.Tensor:
        Chi_mat1_pred = self.forward_mat1(self.y_uniq1)
        Chi1p_mat1_pred, Chi2p_mat1_pred, Chi3p_mat1_pred = Chi_mat1_pred[:, 0:1], Chi_mat1_pred[:, 1:2], Chi_mat1_pred[:, 2:3]

        Chi_mat2_pred = self.forward_mat2(self.y_uniq2)
        Chi1p_mat2_pred, Chi2p_mat2_pred, Chi3p_mat2_pred = Chi_mat2_pred[:, 0:1], Chi_mat2_pred[:, 1:2], Chi_mat2_pred[:, 2:3]

        # Integrals (trapezoidal rule)
        integral_Chi1p_pred1 = torch.trapezoid(Chi1p_mat1_pred, self.y_uniq1, dim=0)
        integral_Chi1p_pred2 = torch.trapezoid(Chi1p_mat2_pred, self.y_uniq2, dim=0)
        null1p1 = torch.zeros(integral_Chi1p_pred1.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        null1p2 = torch.zeros(integral_Chi1p_pred2.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        loss_uniqueness_condition1p = self.loss_fn(integral_Chi1p_pred1, null1p1) + self.loss_fn(integral_Chi1p_pred2, null1p2)

        integral_Chi2p_pred1 = torch.trapezoid(Chi2p_mat1_pred, self.y_uniq1, dim=0)
        integral_Chi2p_pred2 = torch.trapezoid(Chi2p_mat2_pred, self.y_uniq2, dim=0)
        null2p1 = torch.zeros(integral_Chi2p_pred1.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        null2p2 = torch.zeros(integral_Chi2p_pred2.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        loss_uniqueness_condition2p = self.loss_fn(integral_Chi2p_pred1, null2p1) + self.loss_fn(integral_Chi2p_pred2, null2p2)

        integral_Chi3p_pred1 = torch.trapezoid(Chi3p_mat1_pred, self.y_uniq1, dim=0)
        integral_Chi3p_pred2 = torch.trapezoid(Chi3p_mat2_pred, self.y_uniq2, dim=0)
        null3p1 = torch.zeros(integral_Chi3p_pred1.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        null3p2 = torch.zeros(integral_Chi3p_pred2.shape, dtype=DTYPE, requires_grad=False).to(self.device)
        loss_uniqueness_condition3p = self.loss_fn(integral_Chi3p_pred1, null3p1) + self.loss_fn(integral_Chi3p_pred2, null3p2)

        return loss_uniqueness_condition1p + loss_uniqueness_condition2p + loss_uniqueness_condition3p

    def Loss_Collocation(self) -> torch.Tensor:
        Chi_mat1_pred = self.forward_mat1(self.y_mat1)
        Chi1p_mat1_pred, Chi2p_mat1_pred, Chi3p_mat1_pred = Chi_mat1_pred[:, 0:1], Chi_mat1_pred[:, 1:2], Chi_mat1_pred[:, 2:3]
        Chi_mat2_pred = self.forward_mat2(self.y_mat2)
        Chi1p_mat2_pred, Chi2p_mat2_pred, Chi3p_mat2_pred = Chi_mat2_pred[:, 0:1], Chi_mat2_pred[:, 1:2], Chi_mat2_pred[:, 2:3]

        # Derivatives
        # Helper para derivadas primera y segunda evitando código repetido
        def second_derivatives(pred, y):
            grads_first = []
            grads_second = []
            # Cada columna se deriva por separado (salida escalar) para evitar jacobianos grandes
            for col in range(pred.shape[1]):
                comp = pred[:, col:col+1]
                g1 = torch.autograd.grad(comp, y, grad_outputs=torch.ones_like(comp), create_graph=True)[0]
                g2 = torch.autograd.grad(g1, y, grad_outputs=torch.ones_like(g1), create_graph=True)[0]
                grads_first.append(g1)
                grads_second.append(g2)
            return grads_first, grads_second

        (Chi_mat1_pred_y_list, Chi_mat1_pred_yy_list) = second_derivatives(Chi_mat1_pred, self.y_mat1)
        (Chi_mat2_pred_y_list, Chi_mat2_pred_yy_list) = second_derivatives(Chi_mat2_pred, self.y_mat2)

        Chi1p_mat1_pred_y, Chi2p_mat1_pred_y, Chi3p_mat1_pred_y = Chi_mat1_pred_y_list
        Chi1p_mat1_pred_yy, Chi2p_mat1_pred_yy, Chi3p_mat1_pred_yy = Chi_mat1_pred_yy_list
        Chi1p_mat2_pred_y, Chi2p_mat2_pred_y, Chi3p_mat2_pred_y = Chi_mat2_pred_y_list
        Chi1p_mat2_pred_yy, Chi2p_mat2_pred_yy, Chi3p_mat2_pred_yy = Chi_mat2_pred_yy_list

        a = 1
        equ11 = self.R_mat1[5 - a, 5 - a] * Chi1p_mat1_pred_yy + self.R_mat1[5 - a, 4 - a] * Chi2p_mat1_pred_yy + self.R_mat1[5 - a, 3 - a] * Chi3p_mat1_pred_yy
        equ12 = self.R_mat1[4 - a, 5 - a] * Chi1p_mat1_pred_yy + self.R_mat1[4 - a, 4 - a] * Chi2p_mat1_pred_yy + self.R_mat1[4 - a, 3 - a] * Chi3p_mat1_pred_yy
        equ13 = self.R_mat1[3 - a, 5 - a] * Chi1p_mat1_pred_yy + self.R_mat1[3 - a, 4 - a] * Chi2p_mat1_pred_yy + self.R_mat1[3 - a, 3 - a] * Chi3p_mat1_pred_yy

        equ21 = self.R_mat2[5 - a, 5 - a] * Chi1p_mat2_pred_yy + self.R_mat2[5 - a, 4 - a] * Chi2p_mat2_pred_yy + self.R_mat2[5 - a, 3 - a] * Chi3p_mat2_pred_yy
        equ22 = self.R_mat2[4 - a, 5 - a] * Chi1p_mat2_pred_yy + self.R_mat2[4 - a, 4 - a] * Chi2p_mat2_pred_yy + self.R_mat2[4 - a, 3 - a] * Chi3p_mat2_pred_yy
        equ23 = self.R_mat2[3 - a, 5 - a] * Chi1p_mat2_pred_yy + self.R_mat2[3 - a, 4 - a] * Chi2p_mat2_pred_yy + self.R_mat2[3 - a, 3 - a] * Chi3p_mat2_pred_yy

        loss_collocation = (
            nn.functional.mse_loss(equ11, self.null_mat1)
            + nn.functional.mse_loss(equ12, self.null_mat1)
            + nn.functional.mse_loss(equ13, self.null_mat1)
            + nn.functional.mse_loss(equ21, self.null_mat2)
            + nn.functional.mse_loss(equ22, self.null_mat2)
            + nn.functional.mse_loss(equ23, self.null_mat2)
        )
        return loss_collocation

    def Loss_Interface(self) -> torch.Tensor:
        Chi_mat1_pred = self.forward_mat1(self.y_inter)
        Chi1p_mat1_pred, Chi2p_mat1_pred, Chi3p_mat1_pred = Chi_mat1_pred[:, 0:1], Chi_mat1_pred[:, 1:2], Chi_mat1_pred[:, 2:3]
        Chi_mat2_pred = self.forward_mat2(self.y_inter)
        Chi1p_mat2_pred, Chi2p_mat2_pred, Chi3p_mat2_pred = Chi_mat2_pred[:, 0:1], Chi_mat2_pred[:, 1:2], Chi_mat2_pred[:, 2:3]

        Chi1p_mat1_pred_y = torch.autograd.grad(Chi1p_mat1_pred, self.y_inter, grad_outputs=torch.ones_like(Chi1p_mat1_pred), create_graph=True)[0]
        Chi1p_mat2_pred_y = torch.autograd.grad(Chi1p_mat2_pred, self.y_inter, grad_outputs=torch.ones_like(Chi1p_mat2_pred), create_graph=True)[0]
        Chi2p_mat1_pred_y = torch.autograd.grad(Chi2p_mat1_pred, self.y_inter, grad_outputs=torch.ones_like(Chi2p_mat1_pred), create_graph=True)[0]
        Chi2p_mat2_pred_y = torch.autograd.grad(Chi2p_mat2_pred, self.y_inter, grad_outputs=torch.ones_like(Chi2p_mat2_pred), create_graph=True)[0]
        Chi3p_mat1_pred_y = torch.autograd.grad(Chi3p_mat1_pred, self.y_inter, grad_outputs=torch.ones_like(Chi3p_mat1_pred), create_graph=True)[0]
        Chi3p_mat2_pred_y = torch.autograd.grad(Chi3p_mat2_pred, self.y_inter, grad_outputs=torch.ones_like(Chi3p_mat2_pred), create_graph=True)[0]

        a = 1
        equ11 = self.R_mat1[5 - a, 5 - a] * Chi1p_mat1_pred_y + self.R_mat1[5 - a, 4 - a] * Chi2p_mat1_pred_y + self.R_mat1[5 - a, 3 - a] * Chi3p_mat1_pred_y + self.R_mat1[5 - a, self.prob_ind - a]
        equ12 = self.R_mat2[5 - a, 5 - a] * Chi1p_mat2_pred_y + self.R_mat2[5 - a, 4 - a] * Chi2p_mat2_pred_y + self.R_mat2[5 - a, 3 - a] * Chi3p_mat2_pred_y + self.R_mat2[5 - a, self.prob_ind - a]
        equ21 = self.R_mat1[4 - a, 5 - a] * Chi1p_mat1_pred_y + self.R_mat1[4 - a, 4 - a] * Chi2p_mat1_pred_y + self.R_mat1[4 - a, 3 - a] * Chi3p_mat1_pred_y + self.R_mat1[4 - a, self.prob_ind - a]
        equ22 = self.R_mat2[4 - a, 5 - a] * Chi1p_mat2_pred_y + self.R_mat2[4 - a, 4 - a] * Chi2p_mat2_pred_y + self.R_mat2[4 - a, 3 - a] * Chi3p_mat2_pred_y + self.R_mat2[4 - a, self.prob_ind - a]
        equ31 = self.R_mat1[3 - a, 5 - a] * Chi1p_mat1_pred_y + self.R_mat1[3 - a, 4 - a] * Chi2p_mat1_pred_y + self.R_mat1[3 - a, 3 - a] * Chi3p_mat1_pred_y + self.R_mat1[3 - a, self.prob_ind - a]
        equ32 = self.R_mat2[3 - a, 5 - a] * Chi1p_mat2_pred_y + self.R_mat2[3 - a, 4 - a] * Chi2p_mat2_pred_y + self.R_mat2[3 - a, 3 - a] * Chi3p_mat2_pred_y + self.R_mat2[3 - a, self.prob_ind - a]

        loss_interface1 = (
            nn.functional.mse_loss(Chi1p_mat1_pred - Chi1p_mat2_pred, self.null_inter)
            + nn.functional.mse_loss(Chi2p_mat1_pred - Chi2p_mat2_pred, self.null_inter)
            + nn.functional.mse_loss(Chi3p_mat1_pred - Chi3p_mat2_pred, self.null_inter)
        )
        loss_interface2 = (
            nn.functional.mse_loss(equ11 - equ12, self.null_inter)
            + nn.functional.mse_loss(equ21 - equ22, self.null_inter)
            + nn.functional.mse_loss(equ31 - equ32, self.null_inter)
        )
        return loss_interface1 + loss_interface2

    def loss(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_uniqueness_condition = self.Loss_Uniqueness_Condition()
        loss_collocation = self.Loss_Collocation()
        loss_interface = self.Loss_Interface()

        if ADAPTIVE_WEIGHTS:
            loss_total = (
                self.w_collocation * loss_collocation
                + self.w_uniqueness * loss_uniqueness_condition
                + self.w_interface * loss_interface
            )
        else:
            loss_total = loss_collocation + loss_uniqueness_condition + loss_interface
        return loss_total, loss_collocation, loss_uniqueness_condition, loss_interface

    def compute_loss_and_optimize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.optimizer_Adam.zero_grad()
        loss_total, lc, lu, li = self.loss()
        loss_total.backward()
        self.optimizer_Adam.step()
        return loss_total, lc, lu, li

    def Log(self, loss_total, loss_collocation, loss_uniqueness_condition, loss_interface, pbar, it, log_every):
        if loss_total.item() < self.best:
            # Actualizar mejor estado (no guardar aquí)
            self.best = loss_total.item()
            self.best_iter = it
            # Clonar state_dict para evitar mutaciones posteriores
            self.best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        self.loss_log.append(loss_total.item())
        self.loss_uniqueness_condition_log.append(loss_uniqueness_condition.item())
        self.loss_collocation_log.append(loss_collocation.item())
        self.loss_interface_log.append(loss_interface.item())
        
        # Registrar contribuciones ponderadas para verificar balance
        with torch.no_grad():
            self.weighted_collocation_log.append((self.w_collocation * loss_collocation).item())
            self.weighted_uniqueness_log.append((self.w_uniqueness * loss_uniqueness_condition).item())
            self.weighted_interface_log.append((self.w_interface * loss_interface).item())

        if pbar is not None and it % log_every == 0:
            # Calcular contribuciones porcentuales para mostrar balance
            wc = self.weighted_collocation_log[-1]
            wu = self.weighted_uniqueness_log[-1]
            wi = self.weighted_interface_log[-1]
            total_w = wc + wu + wi + 1e-12
            pbar.set_postfix(
                {
                    'Loss': loss_total.item(),
                    'C%': f'{100*wc/total_w:.0f}',  # Porcentaje collocation
                    'U%': f'{100*wu/total_w:.0f}',  # Porcentaje uniqueness
                    'I%': f'{100*wi/total_w:.0f}',  # Porcentaje interface
                }
            )

    def train(self, nIter_Adam=100, nIter_LBFGS=100, nIter_NNCG=0, log_every=100, progress: bool = False, callback=None, callback_every: int = 10,
              epoch_offset: int = 0, start_time_override: float | None = None, quiet: bool = False):
        """Entrena el modelo.
        Args:
            nIter_Adam: iteraciones fase Adam.
            nIter_LBFGS: iteraciones fase L-BFGS.
            nIter_NNCG: iteraciones fase NysNewton-CG.
            log_every: frecuencia de logging interno.
            progress: si True muestra barra de progreso (tqdm); si False, modo silencioso.
            epoch_offset: offset to add to printed iteration numbers (for multi-cycle runs).
            start_time_override: if given, use this as t0 for elapsed-time prints (cumulative across cycles).
            quiet: si True, suprime mensajes de iteración de Adam y LBFGS.
        """
        self.log_every = log_every
        total_iter = nIter_Adam + nIter_LBFGS + nIter_NNCG
        # Allow per-model overrides for print cadences (used for later cycles)
        adam_print_every = int(getattr(self, 'adam_print_every', ADAM_PRINT_EVERY))
        lbfgs_print_every = int(getattr(self, 'lbfgs_print_every', LBFGS_PRINT_EVERY))
        nncg_print_every = int(getattr(self, 'nncg_print_every', NNCG_PRINT_EVERY))
        start_time = start_time_override if start_time_override is not None else time.perf_counter()
        if progress:
            from tqdm import trange
            self.pbar = trange(total_iter)
            iterator = self.pbar
        else:
            self.pbar = None
            iterator = range(total_iter)

        lbfgs_end = nIter_Adam + nIter_LBFGS

        for it in iterator:
            self.it = it
            if it < nIter_Adam:
                loss_total, loss_collocation, loss_uniqueness_condition, loss_interface = self.compute_loss_and_optimize()
                self.Log(loss_total, loss_collocation, loss_uniqueness_condition, loss_interface, self.pbar if progress else None, self.it, self.log_every)
                if ADAPTIVE_WEIGHTS and ((it % 1) == 0):
                    with torch.no_grad():
                        eps = torch.finfo(DTYPE).eps
                        c = loss_collocation.detach() + eps
                        u = loss_uniqueness_condition.detach() + eps
                        i = loss_interface.detach() + eps
                        if self.mae_collocation is None:
                            self.mae_collocation = c
                            self.mae_uniqueness = u
                            self.mae_interface = i
                        beta = torch.tensor(ADAPTIVE_BETA, dtype=DTYPE, device=self.device)
                        self.mae_collocation = beta * self.mae_collocation + (1 - beta) * c
                        self.mae_uniqueness = beta * self.mae_uniqueness + (1 - beta) * u
                        self.mae_interface = beta * self.mae_interface + (1 - beta) * i

                        wc_raw = 1.0 / self.mae_collocation
                        wu_raw = 1.0 / self.mae_uniqueness
                        wi_raw = 1.0 / self.mae_interface

                        geom_mean = (wc_raw * wu_raw * wi_raw) ** (1.0/3.0)
                        scale = 1.0 / (geom_mean + eps)

                        self.w_collocation = (wc_raw * scale).clamp(WEIGHT_MIN, WEIGHT_MAX)
                        self.w_uniqueness = (wu_raw * scale).clamp(WEIGHT_UNIQUENESS_MIN, WEIGHT_MAX)
                        self.w_interface = (wi_raw * scale).clamp(WEIGHT_MIN, WEIGHT_MAX)

                try:
                    if callback is not None and ((it + 1) % callback_every == 0):
                        callback(self, it + 1)
                except Exception:
                    pass
                try:
                    if not quiet and (it + 1) % adam_print_every == 0:
                        elapsed = time.perf_counter() - start_time
                        global_it = it + 1 + epoch_offset
                        print(f"[Time] Adam: elapsed {elapsed:.2f}s after {global_it} iterations (cycle-local {it+1})")
                except Exception:
                    pass
            elif it < lbfgs_end:
                # ── L-BFGS phase ──
                if progress and it == nIter_Adam and not quiet:
                    print("Switching to L-BFGS for fine-tuning...")

                _last_lbfgs = {}

                def closure():
                    self.optimizer_LBFGS.zero_grad()
                    loss_total, loss_collocation, loss_uniqueness_condition, loss_interface = self.loss()
                    loss_total.backward()
                    try:
                        _last_lbfgs['lt'] = loss_total.detach()
                        _last_lbfgs['lc'] = loss_collocation.detach()
                        _last_lbfgs['lu'] = loss_uniqueness_condition.detach()
                        _last_lbfgs['li'] = loss_interface.detach()
                    except Exception:
                        pass
                    return loss_total

                self.optimizer_LBFGS.step(closure)

                try:
                    if all(k in _last_lbfgs for k in ('lt', 'lc', 'lu', 'li')):
                        self.Log(
                            _last_lbfgs['lt'],
                            _last_lbfgs['lc'],
                            _last_lbfgs['lu'],
                            _last_lbfgs['li'],
                            self.pbar if progress else None,
                            self.it,
                            self.log_every,
                        )
                except Exception:
                    pass
                try:
                    if callback is not None and ((it + 1) % callback_every == 0):
                        callback(self, it + 1)
                except Exception:
                    pass
                try:
                    lbfgs_iter = it - nIter_Adam + 1
                    if not quiet and lbfgs_iter > 0 and (lbfgs_iter % lbfgs_print_every == 0):
                        elapsed = time.perf_counter() - start_time
                        global_it = it + 1 + epoch_offset
                        print(f"[Time] L-BFGS: elapsed {elapsed:.2f}s after LBFGS iter {lbfgs_iter} (global it {global_it})")
                except Exception:
                    pass
            else:
                # ── NysNewton-CG phase ──
                if progress and it == lbfgs_end and not quiet:
                    print("Switching to NysNewton-CG for second-order refinement...")

                _last_nncg = {}

                def nncg_closure():
                    # NysNewtonCG computes gradients internally via
                    # autograd.grad – the closure only needs to return
                    # the (graph-connected) scalar loss.
                    loss_total, loss_collocation, loss_uniqueness_condition, loss_interface = self.loss()
                    try:
                        _last_nncg['lt'] = loss_total.detach()
                        _last_nncg['lc'] = loss_collocation.detach()
                        _last_nncg['lu'] = loss_uniqueness_condition.detach()
                        _last_nncg['li'] = loss_interface.detach()
                    except Exception:
                        pass
                    return loss_total

                self.optimizer_NNCG.step(nncg_closure)

                try:
                    if all(k in _last_nncg for k in ('lt', 'lc', 'lu', 'li')):
                        self.Log(
                            _last_nncg['lt'],
                            _last_nncg['lc'],
                            _last_nncg['lu'],
                            _last_nncg['li'],
                            self.pbar if progress else None,
                            self.it,
                            self.log_every,
                        )
                except Exception:
                    pass
                try:
                    if callback is not None and ((it + 1) % callback_every == 0):
                        callback(self, it + 1)
                except Exception:
                    pass
                try:
                    nncg_iter = it - lbfgs_end + 1
                    if not quiet and nncg_iter > 0 and (nncg_iter % nncg_print_every == 0):
                        elapsed = time.perf_counter() - start_time
                        global_it = it + 1 + epoch_offset
                        print(f"[Time] NNCG: elapsed {elapsed:.2f}s after NNCG iter {nncg_iter} (global it {global_it})")
                except Exception:
                    pass

        if self.best_iter < nIter_Adam:
            self.best_iter = self.best_iter
        else:
            self.best_iter = nIter_Adam - 1 + (self.best_iter - nIter_Adam + 1) * self.max_iter_LBFGS

        return self.best_iter, self.loss_log

    def Effective_Properties(self, points1, points2, device: torch.device) -> tuple[float, float, float, float, float, float]:
        y_mat1 = torch.tensor(points1, requires_grad=True, dtype=DTYPE).to(device)
        y_mat2 = torch.tensor(points2, requires_grad=True, dtype=DTYPE).to(device)

        Chi_pred1 = self.forward_mat1(y_mat1)
        Chi1p_mat1_pred, Chi2p_mat1_pred, Chi3p_mat1_pred = Chi_pred1[:, 0:1], Chi_pred1[:, 1:2], Chi_pred1[:, 2:3]
        Chi_pred2 = self.forward_mat2(y_mat2)
        Chi1p_mat2_pred, Chi2p_mat2_pred, Chi3p_mat2_pred = Chi_pred2[:, 0:1], Chi_pred2[:, 1:2], Chi_pred2[:, 2:3]

        Chi1p_mat1_pred_y = torch.autograd.grad(Chi1p_mat1_pred, y_mat1, grad_outputs=torch.ones_like(Chi1p_mat1_pred), create_graph=True)[0]
        Chi1p_mat2_pred_y = torch.autograd.grad(Chi1p_mat2_pred, y_mat2, grad_outputs=torch.ones_like(Chi1p_mat2_pred), create_graph=True)[0]
        Chi2p_mat1_pred_y = torch.autograd.grad(Chi2p_mat1_pred, y_mat1, grad_outputs=torch.ones_like(Chi2p_mat1_pred), create_graph=True)[0]
        Chi2p_mat2_pred_y = torch.autograd.grad(Chi2p_mat2_pred, y_mat2, grad_outputs=torch.ones_like(Chi2p_mat2_pred), create_graph=True)[0]
        Chi3p_mat1_pred_y = torch.autograd.grad(Chi3p_mat1_pred, y_mat1, grad_outputs=torch.ones_like(Chi3p_mat1_pred), create_graph=True)[0]
        Chi3p_mat2_pred_y = torch.autograd.grad(Chi3p_mat2_pred, y_mat2, grad_outputs=torch.ones_like(Chi3p_mat2_pred), create_graph=True)[0]

        def Eff_Coef(y_mat1, y_mat2, i, j=self.prob_ind):
            a = 1
            Exp_mat1 = self.R_mat1[i - a, j - a] + self.R_mat1[i - a, 5 - a] * Chi1p_mat1_pred_y + self.R_mat1[i - a, 4 - a] * Chi2p_mat1_pred_y + self.R_mat1[i - a, 3 - a] * Chi3p_mat1_pred_y
            Exp_mat2 = self.R_mat2[i - a, j - a] + self.R_mat2[i - a, 5 - a] * Chi1p_mat2_pred_y + self.R_mat2[i - a, 4 - a] * Chi2p_mat2_pred_y + self.R_mat2[i - a, 3 - a] * Chi3p_mat2_pred_y
            integral_Exp1 = torch.trapezoid(Exp_mat1, y_mat1, dim=0)
            integral_Exp2 = torch.trapezoid(Exp_mat2, y_mat2, dim=0)
            R_eff = integral_Exp1 + integral_Exp2
            return R_eff

        R1p_eff = Eff_Coef(y_mat1, y_mat2, i=1)
        R2p_eff = Eff_Coef(y_mat1, y_mat2, i=2)
        R3p_eff = Eff_Coef(y_mat1, y_mat2, i=3)
        R4p_eff = Eff_Coef(y_mat1, y_mat2, i=4)
        R5p_eff = Eff_Coef(y_mat1, y_mat2, i=5)
        R6p_eff = Eff_Coef(y_mat1, y_mat2, i=6)

        return (
            R1p_eff.detach().cpu().numpy()[0],
            R2p_eff.detach().cpu().numpy()[0],
            R3p_eff.detach().cpu().numpy()[0],
            R4p_eff.detach().cpu().numpy()[0],
            R5p_eff.detach().cpu().numpy()[0],
            R6p_eff.detach().cpu().numpy()[0],
        )
