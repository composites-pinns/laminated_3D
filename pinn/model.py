from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import time
from .config import (
    DTYPE, NORMALIZE_INPUTS, ADAPTIVE_WEIGHTS, ADAPTIVE_BETA,
    WEIGHT_MIN, WEIGHT_MAX, ADAM_LR, LOG_WEIGHTS
)
from .checkpoint import get_next_version, checkpoint_path_for


class FCN(nn.Module):
    def __init__(self, layers: List[int], device: torch.device, activation: nn.Module | None = None):
        super().__init__()
        self.device = device
        self.layers = layers
        self.activation = activation or nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])
        # Inicialización determinista de pesos (xavier_normal usa la seed global de torch)
        for i in range(len(self.layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        a = X.to(dtype=DTYPE)
        # Optional input normalization to [-1, 1] for y in [0,1]
        if NORMALIZE_INPUTS:
            a = 2.0 * a - 1.0
        for i in range(len(self.layers) - 2):
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
        # Ensure R matrices are tensors on the correct device/dtype to avoid numpy broadcasting issues
        # during graph construction
        self.R_mat1 = torch.tensor(param['R_mat1'], dtype=DTYPE).to(self.device)
        self.R_mat2 = torch.tensor(param['R_mat2'], dtype=DTYPE).to(self.device)

        # NN
        self.layers = layers
        self.best = np.inf
        self.best_state = None  # Almacena el mejor estado (state_dict) para guardado al final de secuencia
        self.activation = nn.Tanh()

        # Networks initialization
        self.fcn_mat1 = FCN(self.layers, self.device, activation=self.activation)
        self.fcn_mat2 = FCN(self.layers, self.device, activation=self.activation)

        self.fcn_mat1.to(self.device, dtype=DTYPE)
        self.fcn_mat2.to(self.device, dtype=DTYPE)

        # AMP/compile desactivados: precisión fija float64

        # Optimizers
        params = list(self.fcn_mat1.parameters()) + list(self.fcn_mat2.parameters())
        self.optimizer_Adam = torch.optim.Adam(params, lr=ADAM_LR)
        self.max_iter_LBFGS = 50
        self.optimizer_LBFGS = torch.optim.LBFGS(
            params,
            max_iter=self.max_iter_LBFGS,
            max_eval=self.max_iter_LBFGS,
            tolerance_grad=1.0 * torch.finfo(DTYPE).eps,
            history_size=50,
        )

        # Losses
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.loss_log = []
        self.loss_collocation_log = []
        self.loss_interface_log = []
        self.loss_uniqueness_condition_log = []
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
        # Compute each term separately (no fused fast path)
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
        # self.optimizer_Adam.step() # OSCAR
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

        if pbar is not None and it % log_every == 0:
            pbar.set_postfix(
                {
                    'Loss': loss_total.item(),
                    'Residual loss': loss_collocation.item(),
                    'Uniqueness condition': loss_uniqueness_condition.item(),
                    'Interface loss': loss_interface.item(),
                }
            )

    def train(self, nIter_Adam=100, nIter_LBFGS=100, log_every=100, progress: bool = False, callback=None, callback_every: int = 10):
        """Entrena el modelo.
        Args:
            nIter_Adam: iteraciones fase Adam.
            nIter_LBFGS: iteraciones fase L-BFGS.
            log_every: frecuencia de logging interno.
            progress: si True muestra barra de progreso (tqdm); si False, modo silencioso.
        """
        self.log_every = log_every
        total_iter = nIter_Adam + nIter_LBFGS
        start_time = time.perf_counter()
        if progress:
            from tqdm import trange
            self.pbar = trange(total_iter)
            iterator = self.pbar
        else:
            self.pbar = None
            iterator = range(total_iter)

        for it in iterator:
            self.it = it
            if it < nIter_Adam:
                loss_total, loss_collocation, loss_uniqueness_condition, loss_interface = self.compute_loss_and_optimize()
                self.optimizer_Adam.step()
                # Adaptive weights update (simple inverse-loss moving average)
                if ADAPTIVE_WEIGHTS and ((it % 1) == 0):
                    with torch.no_grad():
                        # Convert to scalars (avoid zero by adding tiny epsilon)
                        eps = torch.finfo(DTYPE).eps
                        c = loss_collocation.detach() + eps
                        u = loss_uniqueness_condition.detach() + eps
                        i = loss_interface.detach() + eps
                        # Initialize moving averages
                        if self.mae_collocation is None:
                            self.mae_collocation = c
                            self.mae_uniqueness = u
                            self.mae_interface = i
                        # Exponential moving average
                        beta = torch.tensor(ADAPTIVE_BETA, dtype=DTYPE, device=self.device)
                        self.mae_collocation = beta * self.mae_collocation + (1 - beta) * c
                        self.mae_uniqueness = beta * self.mae_uniqueness + (1 - beta) * u
                        self.mae_interface = beta * self.mae_interface + (1 - beta) * i
                        # Target average = mean of MAs
                        avg = (self.mae_collocation + self.mae_uniqueness + self.mae_interface) / 3.0
                        # Inverse proportional weights to bring terms to similar scale
                        wc = (avg / self.mae_collocation).clamp(WEIGHT_MIN, WEIGHT_MAX)
                        wu = (avg / self.mae_uniqueness).clamp(WEIGHT_MIN, WEIGHT_MAX)
                        wi = (avg / self.mae_interface).clamp(WEIGHT_MIN, WEIGHT_MAX)
                        # Normalize weights to keep sum ~ 3
                        s = wc + wu + wi
                        self.w_collocation = 3.0 * wc / s
                        self.w_uniqueness = 3.0 * wu / s
                        self.w_interface = 3.0 * wi / s
                if progress and LOG_WEIGHTS and (it % self.log_every == 0) and ADAPTIVE_WEIGHTS and self.pbar is not None:
                    self.pbar.set_description(f"wC={float(self.w_collocation):.2f}, wU={float(self.w_uniqueness):.2f}, wI={float(self.w_interface):.2f}")
                self.Log(loss_total, loss_collocation, loss_uniqueness_condition, loss_interface, self.pbar if progress else None, self.it, self.log_every)
                # Optional callback for external logging (e.g., relative error checkpoints)
                try:
                    if callback is not None and ((it + 1) % callback_every == 0):
                        callback(self, it + 1)
                except Exception:
                    pass
                # Print elapsed time periodically for long runs
                try:
                    if (it + 1) % 1000 == 0:
                        elapsed = time.perf_counter() - start_time
                        # include current sweeping param/value info if available on self
                        meta_info = ''
                        try:
                            pname = getattr(self, 'sweep_param_name', None)
                            pval = getattr(self, 'sweep_param_value', None)
                            if pname is not None and pval is not None:
                                meta_info = f" para {pname}={pval}"
                        except Exception:
                            meta_info = ''
                        print(f"[Time] Adam: elapsed {elapsed:.2f}s after {it+1} iterations{meta_info}")
                except Exception:
                    pass
            else:
                # Cambio de fase (silencioso si progress=False)
                if progress and it == nIter_Adam:
                    print("Switching to L-BFGS for fine-tuning...")

                def closure():
                    self.optimizer_LBFGS.zero_grad()
                    # AMP desactivado para LBFGS
                    loss_total, loss_collocation, loss_uniqueness_condition, loss_interface = self.loss()
                    loss_total.backward()
                    self.Log(loss_total, loss_collocation, loss_uniqueness_condition, loss_interface, self.pbar if progress else None, self.it, self.log_every)
                    return loss_total

                self.optimizer_LBFGS.step(closure)
                # Invoke callback at LBFGS iterations too (epoch counting continues)
                try:
                    if callback is not None and ((it + 1) % callback_every == 0):
                        callback(self, it + 1)
                except Exception:
                    pass
                # Print elapsed time periodically during LBFGS phase
                try:
                    lbfgs_iter = it - nIter_Adam + 1
                    if lbfgs_iter > 0 and (lbfgs_iter % 100 == 0):
                        elapsed = time.perf_counter() - start_time
                        print(f"[Time] L-BFGS: elapsed {elapsed:.2f}s after LBFGS iter {lbfgs_iter} (global it {it+1})")
                except Exception:
                    pass

        if self.best_iter < nIter_Adam:
            self.best_iter = self.best_iter
        else:
            self.best_iter = nIter_Adam - 1 + (self.best_iter - nIter_Adam + 1) * self.max_iter_LBFGS

        # print("The best model is obtained in EPOCH =" + f"{self.best_iter}.")
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
