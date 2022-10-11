""" Cost functions for the models """

import numpy as np


def calculate_mse(e):
    """ Calculate the MSE for vector e """
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """ Calculate the MAE for vector e """
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """ Calculate the loss """
    e = y - tx.dot(w)
    return calculate_mse(e)
