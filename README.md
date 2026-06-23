# PINN for Laminated Composites

Training Physics-Informed Neural Networks (PINNs) to compute effective elastic properties in elastic laminated composites.

## Project Structure

```
├── pinn_final.py           # Main training script (standard Adam + L-BFGS)
├── pinn_final_nncg.py      # Main training script using NysNewtonCG second-order optimizer
├── hyper_grid.py           # Grid search and hyperparameter sweeps
├── pinn/                   # Core model package
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Hyperparameters and configuration settings
│   ├── data.py             # Collocation point generation
│   ├── device.py           # Hardware detection (CPU/CUDA/MPS)
│   ├── materials.py        # Elastic material properties (Young's modulus, Poisson's ratio)
│   ├── model.py            # PINN architecture, loss calculations, and training loops
│   └── plot_utils.py       # Visualization and plotting utilities
└── optimizer/              # Custom optimizers
    ├── __init__.py         # Package initialization
    └── nys_newton_cg.py    # NysNewtonCG second-order optimizer implementation
```

## Basic Usage

### Standard Training (Adam + L-BFGS)
```bash
# Train all problems (p=1 to p=6)
python pinn_final.py

# Train only a specific problem
python pinn_final.py --target 3

# Force retraining (ignore saved checkpoints)
python pinn_final.py --force-retrain

# Run multi-seed analysis with robust statistics
python pinn_final.py --n-runs 20

# Silence inner iterations (only show cycle-level messages)
python pinn_final.py --quiet-iterations
```

### NysNewtonCG Training (Adam + L-BFGS + NNCG)
```bash
# Train all problems using the second-order NysNewtonCG optimizer
python pinn_final_nncg.py

# Train a specific problem with a convergence tolerance (e.g., 0.01%)
python pinn_final_nncg.py --target 3 --tolerance 0.01

# Choose the stopping criterion: simple (moving average of R_eff), pde-residual, or cv (coefficient of variation)
python pinn_final_nncg.py --stop pde-residual
```

### Grid Search / Hyperparameter Sweep
```bash
# Perform sweeps over epochs, learning rate, and layers
python hyper_grid.py

# Run sweeps in parallel across multiple workers
python hyper_grid.py --parallel --workers 4

# Run a quick sweep using only the first two values of each parameter list
python hyper_grid.py --quick
```

## Main Options

| Argument | Description | Default |
|-----------|-------------|---------|
| `--target N` | Train only problem N (1-6) | all |
| `--targets 1,2,3` | Train a comma-separated list of problems | 1,2,3,4,5,6 |
| `--tolerance X` | Convergence tolerance (%) | 0.05 |
| `--max-cycles N` | Maximum number of Adam + L-BFGS (+ NNCG) cycles | 100 |
| `--force-retrain` | Ignore saved model checkpoints | False |
| `--n-runs N` | Execute N runs with different seeds for robust statistics | 1 |
| `--base-seed S` | Base seed for reproducibility | 1234 |
| `--quiet-iterations` | Suppress training logs within cycles | False |
| `--truncate-pct X` | Percentage of extreme values to trim for the truncated mean | 20 |
| `--stop CRITERION` | Stopping criterion: `simple`, `pde-residual`, or `cv` (NNCG version) | simple |

## Configuration

Edit `pinn/config.py` to modify:
- **Architecture**: `LAYERS = [1, 50, 50, 3]`
- **Epochs per cycle**: `EPOCH_ADAMS`, `EPOCH_LBFGS` (and `EPOCH_NNCG` for the NNCG version)
- **Learning rate**: `ADAM_LR`
- **Material properties**: `E1`, `E2`, `Gamma1`, `Gamma2`
- **Volume fraction**: `VF_mat1`, `VF_mat2`

### Material Contrast Normalization (`NORMALIZE_LOSS_BY_MATERIAL`)

When the elastic contrast $E_1/E_2$ is high ($\ge 10$), the stiffness matrix components of the stiffer material are $10^2 - 10^4$ times larger than those of the softer one. This creates a severe imbalance among the loss terms:

| Loss Component | Typical scale without normalization |
|---|---|
| PDE (collocation) stiff material | $O(C_1^2) \approx 10^8$ |
| PDE (collocation) soft material | $O(C_2^2) \approx 10^4$ |
| Interface traction | $O(C_1^2) \approx 10^8$ |
| Interface continuity | $O(1)$ |
| Uniqueness constraint $\int \chi dy = 0$ | $O(1)$ |

The interface and uniqueness conditions ($O(1)$) become invisible compared to the PDE residuals ($O(10^8)$), preventing the optimizer from finding the correct slopes for the correctors $\chi$.

**Solution**: With `NORMALIZE_LOSS_BY_MATERIAL = True` (enabled by default), both stiffness matrices are divided by a reference factor:

$$C_{\text{ref}} = \max(\|R_{\text{mat1}}\|_F, \|R_{\text{mat2}}\|_F)$$

before constructing the PDE, interface, and uniqueness equations. This yields several desirable properties:
- **All** loss components are simultaneously scaled to $O(1)$.
- The correctors $\chi$ are scale-invariant (the PDE is homogeneous), so the exact solution does not change.
- The final computed effective properties are multiplied back by $C_{\text{ref}}$ to recover physical units (GPa).

Empirical result for $\tilde{C}_{33}$ with $E_1/E_2 = 100$:
- Without normalization: error $\approx 11\%$, systematic negative drift, fails to converge.
- With normalization: **error $< 0.1\%$**, converges in $\approx 15$ cycles.

To disable this normalization (original behavior):
```python
# pinn/config.py
NORMALIZE_LOSS_BY_MATERIAL = False
```

### Subdomain Normalization (`domain_bounds` in FCN)

When the volume fraction $V_1$ is extreme ($\ge 0.7$ or $\le 0.3$), one of the subdomains becomes very narrow. For example, with $V_1 = 0.9$:

| Sub-network | Physical Domain | Normalized Range (*Global* $2y-1$) |
|---|---|---|
| `fcn_mat1` | $[0, 0.9]$ | $[-1, 0.8]$ — range of 1.8 |
| `fcn_mat2` | $[0.9, 1.0]$ | $[0.8, 1.0]$ — range of only **0.2** |

With an input range of only 0.2, `tanh` activations barely vary and the network fails to represent the correctors $\chi$.

**Solution**: Each `FCN` receives the bounds of its own subdomain (`domain_bounds`) and normalizes internally:

$$a = \frac{2 \cdot (y - y_{\text{min}})}{y_{\text{max}} - y_{\text{min}}} - 1$$

This ensures that both sub-networks always see the full input range $[-1, 1]$, regardless of the volume fraction.

Subdomain normalization is applied automatically: `PINN.__init__` uses `VF_mat1` ($= y_I$) to assign `domain_bounds=(0, y_I)` to `fcn_mat1` and `domain_bounds=(y_I, 1)` to `fcn_mat2`.

> [!NOTE]
> Checkpoints trained with the previous global normalization are not compatible and require retraining.

## Results

Results are saved to:
- `final_results/` - Single-run output
- `final_results_multirun/` - Multi-seed statistics output
- `grid_results/` - Grid search sweep output

Generated files:
- `data/summary.csv` - Summary of effective properties
- `data/all_reff_values.json` - Complete set of $R_{\text{eff}}$ values for post-processing
- `images/` - Plots of loss history, error progression, and convergence metrics

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Pandas (for grid sweeps and summaries)
- tqdm (optional, for progress bars)
