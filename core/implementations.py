""" Implementation of the Machine Learning methods """

import logging
import numpy as np

from core.costs import calculate_mae, calculate_mse, compute_loss,\
                        sigmoid, log_likelihood_loss 
import tools.helpers


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
    grad = -tx.T.dot(err)/len(err)
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
        w -= gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info("least_square_Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws


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
        #batch_size=1 as in the requirements
        for y_batch, tx_batch in tools.helpers.batch_iter(y, tx, 1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_LS(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w -= gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        logging.info("least_squares_SGD({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws


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
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        y_pred = sigmoid(tx.dot(w))
        loss = log_likelihood_loss(y, y_pred)
        grad = compute_gradient_LR(tx, y, y_pred)
        # gradient w by descent update
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info("logistic_regression_Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws


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
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        y_pred = sigmoid(tx.dot(w))
        loss = log_likelihood_loss(y, y_pred) + lambda_ * np.sum(w ** 2)
        grad = compute_gradient_LR(tx, y, y_pred) + 2 * lambda_ * w

        # gradient w by descent update
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        logging.info("reg_logistic_regression_Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws

def hyperparams_gs(model_name,cfg, train_data):
    # auxiliary function: I don't delete it since it depends on how you want to refactor the code and on Mirali's implementations but\
    # right now we could just use directly the function get_best_lambda in the trainer. Originally I have used this function because \
    # if we want to generalize the hyperparameters grid search we have different hyperparameters between models. BTW I have done the \
    # degree grid search manually  so now it is basically useless unless it is necessary for Mirali's implementation ( but I think that \
    # we could also use Mirali's learning rate optimization in the trainer)
    if model_name == 'least_squares_GD':
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.01
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])

    elif model_name == 'least_squares_SGD':
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.01
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])

    elif model_name == 'least_squares':
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.01
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])

    elif model_name == 'ridge_regression':
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = get_best_lambda(model_name, train_data)
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])
        

    elif model_name == 'logistic_regression':
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.01
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])
        
        
    elif model_name == 'reg_logistic_regression':
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = get_best_lambda(model_name, train_data)
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])

    return hyperparams        

def get_best_lambda(model_name, train_data, num_folds=5):
    """Grid search with k-fold cross validation on different values of lambda to find the best lambda.

    Args:
        model_name: ztring containing the name of the model we are optimizing lambda for.
        train_data: numpy array of shape (N,D) that contains the training data.
        num_folds: number of fold in which we are dividing the training data to do k-fold cross validation.

    Returns:
        a float that is the best lambda obtained for the analyzed model.
    """
    val_idx, train_ifx = tools.helpers.kfold_cross_validation(train_data["x_train"], num_folds)
    lambdas = np.logspace(-4, 0,20)
    # define lists to store the loss of test data
    # cross validation
    i=0
    logging.info("Running the grid search on lambda!".format(
              bi=i))
    for lambda_ in lambdas:
        logging.info("Evaluating  {bi}/20".format(
              bi=i))
        rmse_test_tmp = []
        rmse_test=[]
        for val_fold, train_fold in zip(val_idx[:num_folds], train_ifx[:num_folds]):
            x_test = np.array(train_data["x_train"])[np.array(val_fold)]
            y_test = np.array(train_data["y_train"])[np.array(val_fold)]
            x_train = np.array(train_data["x_train"])[np.array(train_fold)]
            y_train = np.array(train_data["y_train"])[np.array(train_fold)]
            if model_name == 'ridge_regression':
                _,w = ridge_regression(y_train, x_train, lambda_)
                loss_test = np.sqrt(2 * compute_loss(y_test, x_test, w))
            elif model_name == 'reg_logistic_regression':
                w_initial = np.zeros((x_train.shape[1]))
                gamma=0.1
                max_iters=100
                _,ws= reg_logistic_regression(y_train,x_train, lambda_, w_initial, max_iters, gamma )
                w = ws[-1]
                y_pred = sigmoid(x_test.dot(w))
                loss_test = log_likelihood_loss(y_test, y_pred) + lambda_ * np.sum(w ** 2)
            rmse_test_tmp.append(loss_test)
        rmse_test.append(np.mean(rmse_test_tmp))
        i+=1
    ind_lambda = np.argmin(rmse_test)
    logging.info("Best lambda for model {i} = {x}!".format(
              i=model_name, x=lambdas[ind_lambda]))

    return lambdas[ind_lambda]
