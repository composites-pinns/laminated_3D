# PINN para Compuestos Laminados

Entrenamiento de Physics-Informed Neural Networks para el cálculo de propiedades efectivas en compuestos laminados elásticos.

## Estructura

```
├── pinn_final.py      # Script principal de entrenamiento
└── pinn/              # Módulo con componentes del modelo
    ├── config.py      # Hiperparámetros y configuración
    ├── data.py        # Generación de puntos de colocación
    ├── device.py      # Detección de dispositivo (CPU/CUDA/MPS)
    ├── materials.py   # Propiedades de materiales (módulo de Young, Poisson)
    ├── model.py       # Arquitectura PINN y optimizadores
    └── plot_utils.py  # Utilidades de visualización
```

## Uso Básico

```bash
# Entrenar todos los problemas (p=1 a p=6)
python pinn_final.py

# Entrenar solo un problema específico
python pinn_final.py --target 3

# Forzar reentrenamiento
python pinn_final.py --force-retrain

# Multi-run con estadísticas robustas
python pinn_final.py --n-runs 20

# Silenciar iteraciones internas (solo muestra ciclos)
python pinn_final.py --quiet-iterations
```

## Opciones Principales

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--target N` | Entrenar solo problema N (1-6) | todos |
| `--targets 1,2,3` | Entrenar problemas específicos | 1,2,3,4,5,6 |
| `--tolerance X` | Tolerancia de convergencia (%) | 0.05 |
| `--max-cycles N` | Máximo de ciclos Adam+LBFGS | 100 |
| `--force-retrain` | Ignorar modelos guardados | False |
| `--n-runs N` | Ejecutar N iteraciones con diferentes seeds | 1 |
| `--base-seed S` | Seed base para reproducibilidad | 1234 |
| `--quiet-iterations` | Silenciar mensajes dentro de ciclos | False |
| `--truncate-pct X` | % de valores a eliminar para media truncada | 20 |

## Configuración

Editar `pinn/config.py` para modificar:
- **Arquitectura**: `LAYERS = [1, 50, 50, 3]`
- **Épocas**: `EPOCH_ADAMS`, `EPOCH_LBFGS`
- **Learning rate**: `ADAM_LR`
- **Materiales**: `E1`, `E2`, `Gamma1`, `Gamma2`
- **Fracción de volumen**: `VF_mat1`, `VF_mat2`

## Resultados

Los resultados se guardan en:
- `final_results/` - Ejecución simple
- `final_results_multirun/` - Multi-run con estadísticas

Archivos generados:
- `data/summary.csv` - Resumen de propiedades efectivas
- `data/all_reff_values.json` - Todos los Reff para cálculos posteriores
- `images/` - Gráficas de pérdida y convergencia

## Requisitos

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
