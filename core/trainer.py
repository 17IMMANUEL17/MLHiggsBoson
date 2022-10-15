""" Main logic for the training """

import logging
import numpy as np
from datetime import datetime

from core.implementations import least_squares_GD, logistic_regression, reg_logistic_regression, ridge_regression, \
    least_squares_SGD, least_squares


def run_training(cfg, train_data, test_data):
    """ Running the training """
    logging.info(f'Starting the training!')

    # choose the hyperparameters
    hyperparams = choose_hyperparams(cfg)

    # schedule training for each of the selected models
    for model, to_train in cfg.model_selection.items():
        if to_train:

            # run the training for each set of the hyperparameters
            for i in range(hyperparams.shape[0]):
                # split data into the train / val sets
                x_train, y_train = train_data["x_train"], train_data["y_train"]
                # TODO: Split data into the validation set
                x_val, y_val = None, None

                # initialize the weights
                w_initial = np.zeros((x_train.shape[1]))

                # run the actual training loop
                ws, losses = train(cfg, hyperparams[i, :], model, w_initial, x_train, y_train)

                # run the evaluation
                # accuracy_eval = evaluate(ws, x_val, y_val)


def train(cfg, trial_hyperparams, model_name, w_initial, x_train, y_train):
    """ Running the training loop """
    init_gamma, fin_gamma, gamma_decay, _lambda = trial_hyperparams
    logging.info(f'init_gamma: {init_gamma}; fin_gamma: {fin_gamma}; gamma_decay: {gamma_decay}; _lambda: {_lambda};')
    start_time = datetime.now().replace(microsecond=0)
    logging.info(f'Training the model: {model_name} - started training at {start_time}')

    # select and train the model based on the given model_name
    ws, losses = choose_model(model_name, y_train, x_train, _lambda, w_initial, cfg.n_epochs, init_gamma)

    end_time = datetime.now().replace(microsecond=0)
    execution_time = (end_time - start_time).total_seconds()
    logging.info(f'Training of the model {model_name} finished in {execution_time} seconds \n')
    return ws, losses


def evaluate(ws, x_val, y_val):
    """ Running the evaluation loop """
    # TODO: Implement the model evaluation
    raise NotImplementedError("Evaluation is not implemented yet!")


def choose_hyperparams(cfg):
    """ Choose the hyperparameters and return them as:
        np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]]) """
    if cfg.grid_search:
        logging.info(f'Running the hyperparameter Grid Search!')
        # TODO: Implement the hyperparam search
        hyperparams = None
        raise NotImplementedError("Hyperparam search is not implemented yet!\n")
    else:
        logging.info(f'Default hyperparameters are used for the training!\n')
        initial_gamma = 0.1
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.1
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])
    return hyperparams


def choose_model(model_name, y_train, x_train, lambda_, w_initial, max_iters, gamma):
    """ Choose and train the model based on the given model_name """

    if model_name == 'least_squares_GD':
        w, loss = least_squares_GD(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'least_squares_SGD':
        w, loss = least_squares_SGD(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'least_squares':
        w, loss = least_squares(y_train, x_train)

    elif model_name == 'ridge_regression':
        w, loss = ridge_regression(y_train, x_train, lambda_)

    elif model_name == 'logistic_regression':
        w, loss = logistic_regression(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'reg_logistic_regression':
        w, loss = reg_logistic_regression(y_train, x_train, lambda_, w_initial, max_iters, gamma)

    return w, loss
