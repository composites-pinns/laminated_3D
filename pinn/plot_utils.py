"""Plotting utilities: smoothing and save helpers."""

from typing import List, Iterable
import numpy as np


def smooth_ema(y: Iterable[float], alpha: float = 0.1) -> List[float]:
    """Simple exponential moving average smoothing.
    alpha in (0,1]; smaller alpha -> heavier smoothing.
    """
    out: List[float] = []
    for v in y:
        if not out:
            out.append(float(v))
        else:
            out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def plot_relerr(xs, ys, xlabel: str, title: str, out_path: str, logx: bool = False, smooth_alpha: float | None = None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise

    xs_plot = xs
    fig, ax = plt.subplots(figsize=(7, 4))
    # Mask extreme values (> MAX_VALID) for plotting only (leave CSVs unchanged)
    MAX_VALID = 1e3
    MIN_VALID = 1e-8
    if smooth_alpha is None:
        try:
            _ys = np.array(ys, dtype=float)
            _ys[_ys > MAX_VALID] = np.nan
            _ys[_ys < MIN_VALID] = np.nan
            ax.plot(xs_plot, _ys, linewidth=1.5, label='raw')
        except Exception:
            ys_plot = [float(v) if (isinstance(v, (int, float)) and float(v) <= MAX_VALID) else float('nan') for v in ys]
            ax.plot(xs_plot, ys_plot, linewidth=1.5, label='raw')
    else:
        ys_s = smooth_ema(ys, smooth_alpha)
        try:
            _ys = np.array(ys_s, dtype=float)
            _ys[_ys > MAX_VALID] = np.nan
            _ys[_ys < MIN_VALID] = np.nan
            ax.plot(xs_plot, _ys, linewidth=2.0, label=f'smoothed (alpha={smooth_alpha})')
        except Exception:
            ys_plot = [float(v) if (isinstance(v, (int, float)) and float(v) <= MAX_VALID) else float('nan') for v in ys_s]
            ax.plot(xs_plot, ys_plot, linewidth=2.0, label=f'smoothed (alpha={smooth_alpha})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('RelErr (%)')

    if logx:
        try:
            xs_pos = all(float(x) > 0 for x in xs_plot)
        except Exception:
            xs_pos = False
        if xs_pos:
            ax.set_xscale('log')

    # Only enable log-y when at least one y value is strictly positive
    try:
        ys_pos_any = any(float(y) > 0 for y in ys)
    except Exception:
        ys_pos_any = False
    if ys_pos_any:
        ax.set_yscale('log')

    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reff_evolution(epochs: List, reff_pred: List, reff_exact: float, p_run: int, 
                        out_path: str, smooth_alpha: float | None = None, transitions: List[tuple] | None = None):
    """Grafica la evolución de R_eff predicho vs epochs, con el valor exacto como línea horizontal.
    
    Args:
        epochs: Lista de epochs
        reff_pred: Lista de valores R_eff predichos
        reff_exact: Valor exacto de R_eff (línea horizontal)
        p_run: Índice del problema (1-6)
        out_path: Ruta para guardar el gráfico
        smooth_alpha: Factor de suavizado EMA (None = sin suavizar)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filtrar valores NaN e infinitos
    valid_mask = []
    for v in reff_pred:
        try:
            fv = float(v)
            valid_mask.append(not (np.isnan(fv) or np.isinf(fv)))
        except:
            valid_mask.append(False)
    
    epochs_clean = [e for e, m in zip(epochs, valid_mask) if m]
    reff_clean = [float(v) for v, m in zip(reff_pred, valid_mask) if m]
    
    if not epochs_clean:
        plt.close(fig)
        return
    
    # R_eff predicho
    if smooth_alpha is None:
        ax.plot(epochs_clean, reff_clean, 'b-', linewidth=1.5, 
                label=f'$R^*_{{{p_run}{p_run}}}$ predicho', alpha=0.8)
    else:
        reff_smooth = smooth_ema(reff_clean, smooth_alpha)
        ax.plot(epochs_clean, reff_smooth, 'b-', linewidth=2.0, 
                label=f'$R^*_{{{p_run}{p_run}}}$ predicho (smooth α={smooth_alpha})', alpha=0.9)
    
    # R_eff exacto (línea horizontal roja)
    ax.axhline(y=reff_exact, color='r', linestyle='--', linewidth=2, 
               label=f'$R^*_{{{p_run}{p_run}}}$ exacto = {reff_exact:.4f}')

    # Dibujar líneas verticales en transiciones de fase si se proporcionan
    if transitions:
        drew_adam_lb = False
        drew_lb_adam = False
        for ep, kind in transitions:
            try:
                xval = float(ep)
            except Exception:
                continue
            if kind == 'adam->lbfgs':
                # amarillo
                lab = 'Adam → L-BFGS' if not drew_adam_lb else None
                ax.axvline(x=xval, color='gold', linestyle='--', linewidth=1.5, alpha=0.9, label=lab, zorder=0)
                drew_adam_lb = True
            elif kind == 'lbfgs->adam':
                # verde
                lab = 'L-BFGS → Adam' if not drew_lb_adam else None
                ax.axvline(x=xval, color='limegreen', linestyle='--', linewidth=1.5, alpha=0.9, label=lab, zorder=0)
                drew_lb_adam = True

    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'$R^*_{{{p_run}{p_run}}}$', fontsize=12)
    ax.set_title(f'Evolución de $R^*_{{{p_run}{p_run}}}$ durante el entrenamiento', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Añadir información del error relativo final
    if reff_clean:
        final_pred = reff_clean[-1]
        if reff_exact != 0:
            rel_err = abs(final_pred - reff_exact) / abs(reff_exact) * 100
            ax.text(0.98, 0.02, f'Error relativo final: {rel_err:.2f}%', 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_loss_overlays(loss_csv_paths: Iterable[str], labels: Iterable[str], out_path: str, smooth_alpha: float | None = None):
    """Plot multiple loss CSVs on the same axis.

    Each CSV is expected to have at least an 'epoch' column and either 'loss' or
    'loss_total'. If component columns are present (loss_collocation, ...), they
    are ignored here (use plot_loss_components for per-component visualization).
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        pd = None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise

    fig, ax = plt.subplots(figsize=(7, 4))
    for path, lab in zip(loss_csv_paths, labels):
        try:
            if pd is not None:
                df = pd.read_csv(path)
                if 'epoch' in df.columns:
                    epochs = df['epoch'].tolist()
                else:
                    epochs = list(range(1, len(df) + 1))
                if 'loss_total' in df.columns:
                    losses = df['loss_total'].astype(float).tolist()
                elif 'loss' in df.columns:
                    losses = df['loss'].astype(float).tolist()
                else:
                    # Try to fallback to first numeric column after epoch
                    cols = [c for c in df.columns if c != 'epoch']
                    losses = df[cols[0]].astype(float).tolist() if cols else []
            else:
                import csv
                with open(path, 'r', newline='') as fh:
                    r = csv.reader(fh)
                    headers = next(r)
                    idx_epoch = 0
                    idx_loss = None
                    for i, h in enumerate(headers):
                        if h == 'epoch':
                            idx_epoch = i
                        if h == 'loss_total' or h == 'loss':
                            idx_loss = i
                    epochs = []
                    losses = []
                    for row in r:
                        try:
                            epochs.append(int(row[idx_epoch]))
                        except Exception:
                            epochs.append(len(epochs) + 1)
                        if idx_loss is not None:
                            try:
                                losses.append(float(row[idx_loss]))
                            except Exception:
                                losses.append(float('nan'))
        except Exception:
            continue

        if not losses:
            continue

        # Mask extreme values (> MAX_VALID) for plotting only
        MAX_VALID = 1e3
        MIN_VALID = 1e-12
        if smooth_alpha is None:
            plot_vals = losses
            label = f"v={lab}"
        else:
            plot_vals = smooth_ema(losses, smooth_alpha)
            label = f"v={lab} smoothed"
        try:
            arr = np.array(plot_vals, dtype=float)
            arr[arr > MAX_VALID] = np.nan
            arr[arr < MIN_VALID] = np.nan
            ax.plot(epochs, arr, label=label)
        except Exception:
            plot_vals = [float(v) if (isinstance(v, (int, float)) and float(v) <= MAX_VALID) else float('nan') for v in plot_vals]
            ax.plot(epochs, plot_vals, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Determine whether to use log scale on y by inspecting plotted data
    y_vals = []
    for line in ax.get_lines():
        try:
            y_vals.extend([float(v) for v in line.get_ydata()])
        except Exception:
            continue
    if any(v > 0 for v in y_vals):
        ax.set_yscale('log')

    ax.set_title('Loss overlay')
    # Only show legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_components(loss_csv_path: str, out_path: str, smooth_alpha: float | None = None):
    """Plot loss components (collocation, uniqueness, interface, total) from a single CSV file.

    Si el CSV contiene columnas weighted_*, grafica las contribuciones ponderadas (w*L)
    que suman correctamente al total. Si no, grafica las pérdidas crudas.

    CSV expected columns: epoch, loss_total, weighted_collocation, weighted_uniqueness,
    weighted_interface o loss_collocation, loss_uniqueness, loss_interface.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        pd = None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise

    epochs = []
    loss_total = []
    loss_coll = []
    loss_uniq = []
    loss_int = []
    use_weighted = False  # Flag para saber si usamos contribuciones ponderadas

    if pd is not None:
        try:
            df = pd.read_csv(loss_csv_path)
            if 'epoch' in df.columns:
                epochs = df['epoch'].tolist()
            if 'loss_total' in df.columns:
                loss_total = df['loss_total'].astype(float).tolist()
            elif 'loss' in df.columns:
                loss_total = df['loss'].astype(float).tolist()
            
            # Preferir contribuciones ponderadas si están disponibles
            if 'weighted_collocation' in df.columns:
                loss_coll = df['weighted_collocation'].astype(float).tolist()
                use_weighted = True
            elif 'loss_collocation' in df.columns:
                loss_coll = df['loss_collocation'].astype(float).tolist()
                
            if 'weighted_uniqueness' in df.columns:
                loss_uniq = df['weighted_uniqueness'].astype(float).tolist()
                use_weighted = True
            elif 'loss_uniqueness' in df.columns:
                loss_uniq = df['loss_uniqueness'].astype(float).tolist()
            elif 'loss_uniqueness_condition' in df.columns:
                loss_uniq = df['loss_uniqueness_condition'].astype(float).tolist()
                
            if 'weighted_interface' in df.columns:
                loss_int = df['weighted_interface'].astype(float).tolist()
                use_weighted = True
            elif 'loss_interface' in df.columns:
                loss_int = df['loss_interface'].astype(float).tolist()
        except Exception:
            pass
    else:
        import csv
        try:
            with open(loss_csv_path, 'r', newline='') as fh:
                r = csv.reader(fh)
                headers = next(r)
                idx_epoch = None
                idx_loss = None
                idx_lc = None
                idx_lu = None
                idx_li = None
                # Buscar columnas weighted primero
                for i, h in enumerate(headers):
                    if h == 'epoch':
                        idx_epoch = i
                    if h == 'loss_total' or h == 'loss':
                        idx_loss = i
                    if h == 'weighted_collocation':
                        idx_lc = i
                        use_weighted = True
                    elif h == 'loss_collocation' and idx_lc is None:
                        idx_lc = i
                    if h == 'weighted_uniqueness':
                        idx_lu = i
                        use_weighted = True
                    elif h == 'loss_uniqueness' and idx_lu is None:
                        idx_lu = i
                    elif h == 'loss_uniqueness_condition' and idx_lu is None:
                        idx_lu = i
                    if h == 'weighted_interface':
                        idx_li = i
                        use_weighted = True
                    elif h == 'loss_interface' and idx_li is None:
                        idx_li = i
                for row in r:
                    if idx_epoch is not None:
                        try:
                            epochs.append(int(row[idx_epoch]))
                        except Exception:
                            epochs.append(len(epochs) + 1)
                    if idx_loss is not None:
                        try:
                            loss_total.append(float(row[idx_loss]))
                        except Exception:
                            loss_total.append(float('nan'))
                    if idx_lc is not None:
                        try:
                            loss_coll.append(float(row[idx_lc]))
                        except Exception:
                            loss_coll.append(float('nan'))
                    if idx_lu is not None:
                        try:
                            loss_uniq.append(float(row[idx_lu]))
                        except Exception:
                            loss_uniq.append(float('nan'))
                    if idx_li is not None:
                        try:
                            loss_int.append(float(row[idx_li]))
                        except Exception:
                            loss_int.append(float('nan'))
        except Exception:
            pass

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    MAX_VALID = 1e6  # Aumentado para permitir valores más grandes
    MIN_VALID = 1e-12

    # Helper to prepare array (apply smoothing if requested)
    def _prepare(vals):
        if not vals:
            return []
        arr = None
        try:
            if smooth_alpha is not None:
                vals_proc = smooth_ema(vals, smooth_alpha)
            else:
                vals_proc = [float(v) for v in vals]
            arr = np.array(vals_proc, dtype=float)
            arr[arr > MAX_VALID] = np.nan
            arr[arr < MIN_VALID] = np.nan
            return arr
        except Exception:
            cleaned = [float(v) if (isinstance(v, (int, float)) and float(v) <= MAX_VALID) else float('nan') for v in vals]
            if smooth_alpha is not None:
                try:
                    cleaned = smooth_ema(cleaned, smooth_alpha)
                except Exception:
                    pass
            return np.array(cleaned, dtype=float)

    x_axis = epochs or None
    if loss_total:
        arr = _prepare(loss_total)
        x = x_axis if x_axis is not None else list(range(1, len(arr) + 1))
        ax.plot(x, arr, label='total', color='black', linewidth=1.8)
        plotted = True
    # explicit colors for components to ensure visual distinction
    if loss_coll:
        arr = _prepare(loss_coll)
        x = x_axis if x_axis is not None else list(range(1, len(arr) + 1))
        ax.plot(x, arr, label='collocation', color='#1f77b4')
        plotted = True
    if loss_uniq:
        arr = _prepare(loss_uniq)
        x = x_axis if x_axis is not None else list(range(1, len(arr) + 1))
        ax.plot(x, arr, label='uniqueness', color='#ff7f0e')
        plotted = True
    if loss_int:
        arr = _prepare(loss_int)
        x = x_axis if x_axis is not None else list(range(1, len(arr) + 1))
        ax.plot(x, arr, label='interface', color='#2ca02c')
        plotted = True

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # Enable log-y only if there are positive values
    y_vals = []
    for line in ax.get_lines():
        try:
            y_vals.extend([float(v) for v in line.get_ydata()])
        except Exception:
            continue
    if any(v > 0 for v in y_vals):
        ax.set_yscale('log')

    # Título indica si son contribuciones ponderadas (suman al total) o pérdidas crudas
    title = 'Loss components (weighted, sum=total)' if use_weighted else 'Loss components (raw, may not sum to total)'
    ax.set_title(title)
    if plotted:
        ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
