import numpy as np
from .config import TRAIN_SEED, y_I, N_COLLOCATION, N_UNIQUENESS, N_BOUNDARY, N_POINTS


def set_seeds():
    np.random.seed(TRAIN_SEED)


def build_train_data():
    # Fijar seed explícitamente antes de generar datos aleatorios
    np.random.seed(TRAIN_SEED)
    y_collocation_mat1 = np.random.uniform(0, y_I, N_COLLOCATION)[:, None]
    y_collocation_mat2 = np.random.uniform(y_I, 1, N_COLLOCATION)[:, None]
    assert (y_I not in y_collocation_mat2)

    y_interface = np.array([[y_I]])

    y_uniqueness1 = np.linspace(0, y_I, N_UNIQUENESS)[:, None]
    y_uniqueness2 = np.linspace(y_I, 1, N_UNIQUENESS)[:, None]

    return (
        y_collocation_mat1,
        y_collocation_mat2,
        y_interface,
        y_uniqueness1,
        y_uniqueness2,
    )


def build_eval_points():
    y_points1 = np.linspace(0, y_I, N_POINTS)[:, None]
    y_points2 = np.linspace(y_I, 1, N_POINTS)[:, None]
    return y_points1, y_points2
