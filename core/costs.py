""" Cost functions for the models """

import numpy as np


def calculate_mse(e):
    """Calculate the MSE for vector e"""
    return 1 / 2 * np.mean(e**2)


def calculate_mae(e):
    """Calculate the MAE for vector e"""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss"""
    e = y - tx.dot(w)
    return calculate_mse(e)


def sigmoid(x):
    """Apply sigmoid function on x."""
    return 1 / (1 + np.exp(-x))


def log_likelihood_loss(y_true, y_pred):
    """Compute negative log likelihood for logistic regression

    Args:
        y_true: true labels, numpy array of shape (N,)
        y_pred: predicted labels, numpy array of shape (N,)

    Returns:
        loss: negative log likelihood, scalar
    """
    # binary cross entropy
    y_true.shape = (y_true.shape[0], 1)
    y_pred.shape = (y_pred.shape[0], 1)
    y_zero_loss = y_true * np.log(y_pred + 1e-15)
    y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-15)
    return -np.sum(y_zero_loss + y_one_loss) / len(y_true)
