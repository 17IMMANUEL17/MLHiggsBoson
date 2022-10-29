""" Implementation of the Machine Learning methods """

import logging
import numpy as np

from core.costs import calculate_mae, calculate_mse, compute_loss, sigmoid, log_likelihood_loss
from core.linesearch import BFGS
from tools.helpers import batch_iter


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


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters 
            as numpy arrays of shape (D, ), for each iteration of GD
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient_LS(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info("Least Squares GD ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # batch_size=1 is a project requirement
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_LS(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        logging.info("Least Squares SGD ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    mse = compute_loss(y, tx, w)
    return mse, w


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return loss, w


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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD (y ∈ {0, 1})

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters
            as numpy arrays of shape (D, ), for each iteration of GD
    """
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
        logging.info("Logistic Regression GD ({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent
        or SGD (y ∈ {0, 1}, with regularization term λ||w||^2)

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization term
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters
            as numpy arrays of shape (D, ), for each iteration of GD
    """
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
        logging.info("Reg Logistic Regression GD ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def logistic_regression_bfgs(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using bfgs algorithm (y ∈ {0, 1})

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters
            as numpy arrays of shape (D, ), for each iteration of GD
    """
    bfgs = BFGS(initial_w.shape[0], 0.5, 0.1)
    step_function = bfgs.step
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        w, loss = step_function(w, tx, y, gamma)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info("Logistic Regression BFGS ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def reg_logistic_regression_bfgs(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using bfgs algorithm (y ∈ {0, 1}, with regularization term λ||w||^2)

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization term
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters
            as numpy arrays of shape (D, ), for each iteration of GD
    """
    bfgs = BFGS(initial_w.shape[0], 0.5, 0.1)
    step_function = bfgs.step
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        w, loss = step_function(w, tx, y, gamma, lambda_=lambda_)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info("Reg Logistic Regression BFGS({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w


def prepare_gs(init_gamma, final_gamma, gamma_decay, _lambda, poly_degree):
    """ Prepare the grid search hyperparam matrix """
    dimensions = len(init_gamma) * len(final_gamma) * len(gamma_decay) * len(_lambda) * len(poly_degree)
    a, b, c, d, e = np.meshgrid(init_gamma, final_gamma, gamma_decay, _lambda, poly_degree)
    a, b, c, d, e = np.reshape(a, (dimensions, -1)), np.reshape(b, (dimensions, -1)), np.reshape(c, (dimensions, -1)), \
                    np.reshape(d, (dimensions, -1)), np.reshape(e, (dimensions, -1))
    hyperparam_matrix = np.concatenate((a, b, c, d, e), axis=1)
    return hyperparam_matrix
