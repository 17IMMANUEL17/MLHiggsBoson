""" Implementation of the Machine Learning methods """

import logging

import numpy as np

from core.costs import compute_loss, log_likelihood_loss, sigmoid
from tools.helpers import batch_iter, compute_gradient_LS, logistic_regression_GD_step


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy array of shape (D, ) containing the model parameters obtained by the last iteration of GD
        loss: a float containing the loss value (scalar) associated to w
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient_LS(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info(
            "Least Squares Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy array of shape (D, ) containing the model parameters obtained by the last iteration of SGD
        loss: a float containing the loss value (scalar) associated to w
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # batch_size=1 is a project requirement
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_LS(y_batch, tx_batch, w)
            # calculate loss and update weights
            loss = compute_loss(y, tx, w)
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
        logging.info(
            "Least Squares SGD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: a float containing the loss value (scalar) associated to w
    """
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: a numpy array of shape (D, ) containing the model parameters
        loss: a float containing the loss value (scalar) associated to w
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1})

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy array of shape (D, ) containing the model parameters
        loss: a float containing the loss value (scalar) for the last iteration of GD
    """
    initial_w.shape = (initial_w.shape[0], 1)
    y.shape = (y.shape[0], 1)
    step_function = logistic_regression_GD_step
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        w, loss = step_function(w, tx, y, gamma)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info(
            "Logistic Regression GD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    loss = log_likelihood_loss(y, sigmoid(tx.dot(w)))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
        or SGD (y ∈ {0, 1}, with regularization term λ||w||^2)

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization term
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy array of shape (D, ) containing the model parameters
        loss: a float containing the loss value (scalar) for the last iteration of GD
    """
    initial_w.shape = (initial_w.shape[0], 1)
    y.shape = (y.shape[0], 1)
    step_function = logistic_regression_GD_step
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        w, loss = step_function(w, tx, y, gamma, lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info(
            "Reg Logistic Regression GD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    loss = log_likelihood_loss(y, sigmoid(tx.dot(w)))
    return w, loss
