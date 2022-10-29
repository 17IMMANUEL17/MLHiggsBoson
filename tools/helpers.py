""" Helper functions for various purposes """

import numpy as np
from core.costs import log_likelihood_loss, sigmoid


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def batch_iter(y, tx, batch_size, num_batches, shuffle=True):
    """Iterate through data in batches.
    Args:
        tx (np.ndarray): Features data
        y (np.ndarray): Labels data
        batch_size (int, optional): Batch size. Defaults to None (i.e. full batch)
        num_batches (int, optional): Number of batches to iterate through. Defaults to None (i.e. use all data)
        shuffle (bool, optional): Whether to shuffle the data before generating batches. Defaults to True.
    Yields:
        Generator: (tx_batch, y_batch)
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        y = y[shuffle_indices]
        tx = tx[shuffle_indices]

    batch_size = batch_size or len(tx)
    batches = int(np.ceil(len(tx) // batch_size))
    num_batches = num_batches or batches

    for i in range(num_batches):
        yield y[i * batch_size:(i + 1) * batch_size], tx[i * batch_size:(i + 1) * batch_size]


def kfold_cross_validation(data, folds_num=5):
    """ Divides all data samples into groups in order to use the whole dataset for training and validation """
    data_idx = np.arange(data.shape[0])
    test_idx, train_idx = [], []

    # size of each fold
    fold_size = int(data.shape[0] / folds_num)
    for i in range(folds_num):
        test_fold = data_idx[i * fold_size: (i + 1) * fold_size]
        train_fold = np.delete(data_idx, test_fold)

        test_idx.append(test_fold)
        train_idx.append(train_fold)

    return test_idx, train_idx


def prepare_gs(init_gamma, final_gamma, gamma_decay, _lambda, poly_degree):
    """ Prepare the grid search hyperparam matrix """
    dimensions = len(init_gamma) * len(final_gamma) * len(gamma_decay) * len(_lambda) * len(poly_degree)
    a, b, c, d, e = np.meshgrid(init_gamma, final_gamma, gamma_decay, _lambda, poly_degree)
    a, b, c, d, e = np.reshape(a, (dimensions, -1)), np.reshape(b, (dimensions, -1)), np.reshape(c, (dimensions, -1)), \
                    np.reshape(d, (dimensions, -1)), np.reshape(e, (dimensions, -1))
    hyperparam_matrix = np.concatenate((a, b, c, d, e), axis=1)
    return hyperparam_matrix


def compute_gradient_LS(y, tx, w):
    """Computes the gradient of the MSE at w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def compute_gradient_LR(tx, y_true, y_pred):
    """
    Compute the gradient of loss.
    Args:
        tx: numpy array of shape (N,D), D is the number of features.
        y_true: true labels, numpy array of shape (N,)
        y_pred: predicted labels, numpy array of shape (N,)
    Returns:
        grad: gradient of loss, numpy array of shape (D,)
    """
    return tx.T.dot(y_pred - y_true) / len(y_true)


def logistic_regression_GD_step(w, tx, y, gamma, lambda_=0):
    """One step of Gradient Descent for Logistic Regression.

    Args:
        w: numpy array of shape (D,), D is the number of features.
        tx: numpy array of shape (N,D), N is the number of samples.
        y: numpy array of shape (N,), true labels.
        gamma: scalar.

    Returns:
        grad: gradient of loss, numpy array of shape (D,)
    """
    # compute loss, gradient
    y_pred = sigmoid(tx.dot(w))
    grad = compute_gradient_LR(tx, y, y_pred) + 2 * lambda_ * w
    loss = log_likelihood_loss(y, y_pred, w, lambda_)

    # gradient w by descent update
    w = w - gamma * grad

    return w, loss
