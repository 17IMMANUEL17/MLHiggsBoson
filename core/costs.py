""" Cost functions for the models """

import numpy as np


def calculate_mse(e):
    """ Calculate the MSE for vector e """
    return 1 / (2*len(e))*e.dot(e)


def calculate_mae(e):
    """ Calculate the MAE for vector e """
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """ Calculate the loss """
    e = y - tx.dot(w)
    return calculate_mse(e)


def sigmoid(x):
    """ Apply sigmoid function on x."""
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
    y_zero_loss = y_true * np.log(y_pred + 1e-9)
    y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)
