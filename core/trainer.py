""" Main logic for the training """

import logging
import numpy as np
from datetime import datetime

from core.implementations import least_squares_GD,\
        logistic_regression, reg_logistic_regression


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
                accuracy_eval = evaluate(ws, x_val, y_val)


def train(cfg, trial_hyperparams, model_name, w_initial, x_train, y_train):
    """ Running the training loop """
    init_gamma, fin_gamma, gamma_decay, _lambda = trial_hyperparams
    logging.info(f'init_gamma: {init_gamma}; fin_gamma: {fin_gamma}; gamma_decay: {gamma_decay}; _lambda: {_lambda};')
    start_time = datetime.now().replace(microsecond=0)
    logging.info(f'Training the model: {model_name} - started training at {start_time} for {cfg.n_epochs} epochs\n')

    # TODO: Implement intelligent model selection based on the given model name
    # model = least_squares_GD
    model = logistic_regression

    # train the model
    ws, losses = model(y=y_train,
                       tx=x_train,
                       initial_w=w_initial,
                       max_iters=cfg.n_epochs,
                       gamma=init_gamma)

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
        raise NotImplementedError("Hyperparam search is not implemented yet!")
    else:
        logging.info(f'Default hyperparameters are used for the training!')
        initial_gamma = 0.1
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.1
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])
    return hyperparams
