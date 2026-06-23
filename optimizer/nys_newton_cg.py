"""
NysNewtonCG – A PyTorch optimizer implementing the Nyström-approximated
Newton-CG method.

Reference:
  Rathore, Berahas, Brust & Gower,
  "Challenges in Training PINNs: A Loss Landscape Perspective" /
  "NysNewtonCG: Convergence Analysis of an Approximate Newton-CG Method"
  (2024).

Algorithm outline (cf. Appendix E of the paper):
  At each step k:
    1. Compute the gradient  g = ∇f(θ).
    2. Build a rank-r Nyström approximation of the Hessian:
         • Draw a random Gaussian sketch  Ω ∈ ℝ^{n×r}.
         • Evaluate  Y = H Ω  via Hessian-vector products.
         • Form  Ŷ = Y + μ Ω   ( μ = regularisation shift ).
         • Cholesky:  C = chol(Ωᵀ Ŷ)   (r × r).
         • Û = Ŷ C⁻¹.
         • The Nyström Hessian proxy is
              H̃ = Û Ûᵀ +  μ I        (never formed explicitly).
    3. Solve  H̃ d = −g  for the Newton direction d via CG
       (using Woodbury for mat-vecs).
    4. (Optional) back-tracking line search.
    5. Update  θ ← θ + α d.

The implementation works entirely with the **flat parameter vector** so it
is compatible with any ``nn.Module``.  Hessian-vector products are computed
with ``torch.autograd.functional.hvp`` (or the more efficient
``torch.func.hessian_vector_product`` when available).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch.optim import Optimizer


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _flat_grad(params):
    """Concatenate gradients of all parameters into a single 1-D tensor."""
    gs = []
    for p in params:
        if p.grad is not None:
            gs.append(p.grad.detach().reshape(-1))
        else:
            gs.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
    return torch.cat(gs)


def _set_flat_params(params, flat):
    """Write *flat* back into the parameter tensors (in-place)."""
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].reshape(p.shape))
        offset += n


def _get_flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


# ──────────────────────────────────────────────────────────────────────────────
# Hessian-vector product via autograd
# ──────────────────────────────────────────────────────────────────────────────
def _hvp_autograd(grad_params, params, vec):
    """
    Compute Hessian-vector product H·v  given  grad_params = ∇f(θ)
    (which must have been computed with ``create_graph=True``).

    Returns a 1-D tensor of the same length as *vec*.
    """
    # grad_params is a list/tuple of gradient tensors (one per param)
    # We split vec into pieces matching each parameter
    pieces = []
    offset = 0
    for p in params:
        n = p.numel()
        pieces.append(vec[offset:offset + n].reshape(p.shape))
        offset += n

    hvp_tuple = torch.autograd.grad(
        grad_params, params, grad_outputs=pieces,
        retain_graph=True, allow_unused=True,
    )
    parts = []
    for h, p in zip(hvp_tuple, params):
        if h is None:
            parts.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        else:
            parts.append(h.reshape(-1))
    return torch.cat(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Nyström preconditioned CG
# ──────────────────────────────────────────────────────────────────────────────
def _nystrom_pcg(hvp_fn, g, U, mu, cg_tol, max_cg_iters):
    """
    Solve  (H + μI) d = -g  approximately via CG, where
    H is accessed only through hvp_fn,  and the Nyström preconditioner is:

        M⁻¹ = (1/μ) I  −  (1/μ²) U (I + (1/μ) UᵀU)⁻¹ Uᵀ

    (Woodbury inverse of  Û Ûᵀ + μ I).

    Returns d (1-D tensor).
    """
    n = g.shape[0]
    device = g.device
    dtype = g.dtype

    # Precompute factorisation for preconditioner application
    # M = U Uᵀ + μ I   →   M⁻¹ v  via Woodbury
    # (1/μ)(v - U (μ I_r + UᵀU)^{-1} Uᵀ v)
    r = U.shape[1] if U.ndim == 2 else 0
    if r > 0:
        UtU = U.T @ U  # r×r
        inner = mu * torch.eye(r, dtype=dtype, device=device) + UtU
        # Cholesky for stable solve
        try:
            L_inner = torch.linalg.cholesky(inner)
            use_chol = True
        except Exception:
            use_chol = False
    else:
        use_chol = False

    def precond(v):
        """Apply M⁻¹ v."""
        if r == 0:
            return v / mu
        Utv = U.T @ v  # r
        if use_chol:
            sol = torch.cholesky_solve(Utv.unsqueeze(1), L_inner).squeeze(1)
        else:
            sol = torch.linalg.solve(inner, Utv)
        return (v - U @ sol) / mu

    # CG  on  A d = b  where  A = H + μI,  b = -g
    b = -g
    x = torch.zeros(n, dtype=dtype, device=device)
    residual = b.clone()  # r0 = b - A·0 = b
    z = precond(residual)
    p = z.clone()
    rz = residual.dot(z)
    rz0 = rz.clone()

    for _ in range(max_cg_iters):
        Ap = hvp_fn(p) + mu * p  # (H + μI) p
        pAp = p.dot(Ap)
        if pAp <= 0:
            # Negative curvature — fall back to steepest descent direction
            if x.norm() < 1e-12:
                return -g  # steepest descent
            return x
        alpha = rz / pAp
        x = x + alpha * p
        residual = residual - alpha * Ap
        if residual.norm() <= cg_tol * b.norm():
            break
        z_new = precond(residual)
        rz_new = residual.dot(z_new)
        beta = rz_new / rz
        p = z_new + beta * p
        rz = rz_new

    return x


# ──────────────────────────────────────────────────────────────────────────────
# The Optimizer
# ──────────────────────────────────────────────────────────────────────────────
class NysNewtonCG(Optimizer):
    """Nyström-approximated Newton-CG optimizer for PyTorch.

    Args:
        params: iterable of parameters to optimize.
        lr: step-size / learning rate (default 1.0).
        rank: rank *r* of the Nyström sketch (default 10).
        mu: Tikhonov regularisation added to the Hessian (default 1e-4).
        cg_tol: relative residual tolerance for CG (default 1e-5).
        max_cg_iters: maximum CG iterations (default 50).
        line_search: whether to perform Armijo back-tracking (default True).
        ls_alpha_init: initial trial step for line search (default 1.0).
        ls_c: sufficient-decrease parameter (default 1e-4).
        ls_rho: back-tracking factor (default 0.5).
        ls_max_iters: maximum back-tracking iterations (default 20).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        rank: int = 10,
        mu: float = 1e-4,
        cg_tol: float = 1e-5,
        max_cg_iters: int = 50,
        line_search: bool = True,
        ls_alpha_init: float = 1.0,
        ls_c: float = 1e-4,
        ls_rho: float = 0.5,
        ls_max_iters: int = 20,
    ):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.rank = rank
        self.mu = mu
        self.cg_tol = cg_tol
        self.max_cg_iters = max_cg_iters
        self.line_search = line_search
        self.ls_alpha_init = ls_alpha_init
        self.ls_c = ls_c
        self.ls_rho = ls_rho
        self.ls_max_iters = ls_max_iters

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Callable):
        """
        Perform a single NysNewton-CG step.

        ``closure`` must return the scalar loss (do **not** call backward).
        The optimizer computes gradients internally via ``torch.autograd.grad``
        with ``create_graph=True`` to enable Hessian-vector products,
        avoiding the reference-cycle memory leak that
        ``loss.backward(create_graph=True)`` would cause.
        """
        # ── 0. Collect all trainable parameters ──────────────────────
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    all_params.append(p)
        lr = self.param_groups[0]['lr']
        device = all_params[0].device
        dtype = all_params[0].dtype
        n = sum(p.numel() for p in all_params)

        # ── 1. Forward pass & graph-connected gradients ──────────────
        with torch.enable_grad():
            loss = closure()  # forward only – no backward!

            # Compute gradients *with* graph so we can differentiate again
            # for Hessian-vector products.
            grad_params = torch.autograd.grad(
                loss, all_params, create_graph=True, allow_unused=True,
            )
            grad_params = [
                gp if gp is not None else torch.zeros_like(p)
                for gp, p in zip(grad_params, all_params)
            ]

        # Detached flat gradient vector for linear-algebra operations
        g = torch.cat([gp.detach().reshape(-1) for gp in grad_params])

        # ── 2. Nyström sketch ────────────────────────────────────────
        rank = min(self.rank, n)
        Omega = torch.randn(n, rank, dtype=dtype, device=device)
        Omega, _ = torch.linalg.qr(Omega)

        # Hessian-vector products  Y = H Ω   (column by column)
        Y = torch.zeros_like(Omega)
        with torch.enable_grad():
            for j in range(rank):
                Y[:, j] = _hvp_autograd(grad_params, all_params, Omega[:, j])

        # Shifted sketch
        Y_hat = Y + self.mu * Omega  # (n, r)

        # ── Robust factorisation of  Ωᵀ Ŷ  (r × r) ─────────────────
        OtY = Omega.T @ Y_hat
        OtY = 0.5 * (OtY + OtY.T)  # symmetrise

        C = _robust_cholesky(OtY, dtype, device, rank)

        if C is not None:
            U_hat = torch.linalg.solve_triangular(C, Y_hat.T, upper=False).T
        else:
            # Eigendecomposition fallback – always succeeds
            U_hat = _nystrom_via_eigh(OtY, Y_hat, Omega, self.mu, dtype, device)

        # ── 3. CG solve ─────────────────────────────────────────────
        def hvp_fn(v):
            with torch.enable_grad():
                return _hvp_autograd(grad_params, all_params, v)

        direction = _nystrom_pcg(
            hvp_fn, g, U_hat, self.mu,
            self.cg_tol, self.max_cg_iters,
        )

        # ── 4. Line search (Armijo back-tracking) ───────────────────
        theta0 = _get_flat_params(all_params)
        f0 = float(loss)
        slope = g.dot(direction)

        alpha = self.ls_alpha_init * lr
        if self.line_search and slope < 0:
            for _ in range(self.ls_max_iters):
                _set_flat_params(all_params, theta0 + alpha * direction)
                with torch.enable_grad():
                    trial_loss = closure()
                f_trial = float(trial_loss)
                if f_trial <= f0 + self.ls_c * alpha * float(slope):
                    break
                alpha *= self.ls_rho
        else:
            _set_flat_params(all_params, theta0 + alpha * direction)

        # ── 5. Cleanup: break reference cycles ───────────────────────
        del grad_params, loss
        for p in all_params:
            p.grad = None

        return f0


# ──────────────────────────────────────────────────────────────────────────────
# Robust Cholesky helpers
# ──────────────────────────────────────────────────────────────────────────────
def _robust_cholesky(M, dtype, device, r):
    """
    Try Cholesky with progressively larger diagonal shifts.
    Returns the lower-triangular factor L, or None if all attempts fail.
    """
    shifts = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    I_r = torch.eye(r, dtype=dtype, device=device)
    for s in shifts:
        try:
            return torch.linalg.cholesky(M + s * I_r)
        except Exception:
            continue
    return None


def _nystrom_via_eigh(OtY, Y_hat, Omega, mu, dtype, device):
    """
    Eigendecomposition fallback for the Nyström factor when Cholesky
    fails even with large shifts (strong indefiniteness).

    Clamps eigenvalues to be >= mu so the preconditioner remains SPD.
    Returns U_hat  (n, r).
    """
    eigvals, eigvecs = torch.linalg.eigh(OtY)
    eigvals = eigvals.clamp(min=mu)
    inv_sqrt = 1.0 / torch.sqrt(eigvals)
    U_hat = Y_hat @ (eigvecs * inv_sqrt.unsqueeze(0))
    return U_hat
