    import numpy as np
import pandas as pd


def mse(y_true, y_pred, lower_bound, upper_bound):
    mask = np.max(y_true, axis=2) > upper_bound
    if np.count_nonzero(mask):
        return np.mean((y_true[mask] - y_pred[mask]) ** 2)
    else:
        return 0


def rmse(y_true, y_pred, lower_bound, upper_bound):
    mask = np.max(y_true, axis=2) > upper_bound
    if np.count_nonzero(mask):
        diff = np.mean((y_true[mask] - y_pred[mask]) ** 2, axis=1) ** 0.5
        return np.mean(diff)
    else:
        return 0


def mange(y_true, y_pred, lower_bound, upper_bound):
    mask = np.max(y_true, axis=2) > upper_bound
    if np.count_nonzero(mask):
        eps = 1e-10
        true_norms = np.linalg.norm(y_true[mask], axis=1)
        pred_norms = np.linalg.norm(y_pred[mask], axis=1)
        product = (y_true[mask] * y_pred[mask]).sum(axis=1)
        product /= true_norms
        product /= pred_norms
        product -= eps
        return np.mean(np.arccos(product))
    else:
        return 0


def nse(y_true, y_pred, lower_bound, upper_bound):
    mask = np.max(y_true, axis=2) > upper_bound
    if np.count_nonzero(mask):
        true_norms = np.linalg.norm(y_true[mask], axis=1, ord=1)
        pred_norms = np.linalg.norm(y_pred[mask], axis=1, ord=1)
        diff = np.linalg.norm(y_pred[mask] - y_true[mask], axis=1, ord=1)
        return np.mean(diff / true_norms / pred_norms)
    else:
        return 0


def coef_prop(a, b, lower_bound, upper_bound, base_coef=10):
    """returns argmin_x |b - ax| in L2 sense"""

    m1 = np.max(a, axis=2) < upper_bound
    m2 = np.min(a, axis=2) > lower_bound
    m3 = np.max(b, axis=2) < upper_bound
    m4 = np.min(b, axis=2) > lower_bound

    if np.count_nonzero(m1 & m2):
        a = a[m1 & m2 & m3 & m4, :].flatten()
        b = b[m1 & m2 & m3 & m4, :].flatten()  # |b - ax| -> min on x
        coef_prop = (a.dot(b)) / (a.dot(a))
    else:
        coef_prop = base_coef

    return coef_prop
