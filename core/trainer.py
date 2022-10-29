""" Main logic for the training """

import logging
import os
from datetime import datetime

import numpy as np

from core.costs import sigmoid
from core.supplementary_implementations import (logistic_regression_bfgs,
                                                reg_logistic_regression_bfgs)
from implementations import (least_squares, logistic_regression,
                             mean_squared_error_gd, mean_squared_error_sgd,
                             reg_logistic_regression, ridge_regression)
from tools.cfg_parser import Config
from tools.helpers import build_poly, kfold_cross_validation, prepare_gs
from tools.utils import create_submission


def run_training(cfg, train_data, test_data, num_folds=5):
    """ Running the training """
    logging.info(f'Starting the training!')

    # choose the hyperparameters
    hyperparams = choose_hyperparams(cfg)

    # schedule training for each of the selected models
    for model, to_train in cfg.model_selection.items():
        if to_train:

            # run the training for each set of the hyperparameters
            hyperparam_acc, hyperparam_pred = [], []
            for i in range(hyperparams.shape[0]):
                acc, pred = [], []

                # generate indexes of data for k-fold cross validation for train / val split
                val_idx, train_ifx = kfold_cross_validation(train_data["x_train"], num_folds)
                fold = 1 if not cfg.cross_validation else num_folds

                # run the training for each fold in the cross validation
                for val_fold, train_fold in zip(val_idx[:fold], train_ifx[:fold]):

                    # get validation data from the current fold
                    x_val = np.array(train_data["x_train"])[np.array(val_fold)]
                    y_val = np.array(train_data["y_train"])[np.array(val_fold)]

                    # get training data from the current fold
                    x_train = np.array(train_data["x_train"])[np.array(train_fold)]
                    y_train = np.array(train_data["y_train"])[np.array(train_fold)]

                    # get test data
                    x_test = np.array(test_data['x_test'])

                    # apply the polynomial feature extension
                    polynomial_degree = int(hyperparams[i, -1])
                    if cfg.polynomial_features:
                        x_train = build_poly(x_train, polynomial_degree)
                        x_val = build_poly(x_val, polynomial_degree)
                        x_test = build_poly(x_test, polynomial_degree)

                    # initialize the weights
                    w_initial = np.zeros((x_train.shape[1]))

                    # run the actual training loop
                    w, loss = train(cfg, fold, hyperparams[i, :], model, w_initial, x_train, y_train)

                    # run evaluation on the val data
                    accuracy_eval = evaluate(model, w, x_val, y_val)
                    acc.append(accuracy_eval)
        
                    # run evaluation on the test data
                    if 'logistic_regression' in model:
                        pred_test = np.round(sigmoid(x_test.dot(w.reshape(-1, 1))))
                    else:
                        pred_test = np.round(x_test.dot(w))
                    pred.append(pred_test)

                # store the results to get the best hyperparameters
                hyperparam_acc.append(np.mean(acc))
                hyperparam_pred.append(np.round(np.mean(np.array(pred), axis=0)))

            # select the best model (the best hyperparam setting) and create the .csv submission
            choose_best_result(cfg, hyperparams, hyperparam_acc, hyperparam_pred, model, loss, test_data)


def train(cfg, fold, trial_hyperparams, model_name, w_initial, x_train, y_train):
    """ Running the training loop """
    init_gamma, fin_gamma, gamma_dec, _lambda, poly_degree = trial_hyperparams
    logging.info(f'init_gamma: {init_gamma}; '
                 f'fin_gamma: {fin_gamma}; '
                 f'gamma_decay: {gamma_dec}; '
                 f'_lambda: {_lambda}; '
                 f'poly_degree: {int(poly_degree)};')

    start_time = datetime.now().replace(microsecond=0)
    logging.info(f'Training the model: {model_name} for fold {fold} - started training at {start_time}')

    # select and train the model based on the given model_name
    w, loss = choose_model(model_name, y_train, x_train, _lambda, w_initial, cfg.n_epochs, init_gamma)

    end_time = datetime.now().replace(microsecond=0)
    execution_time = (end_time - start_time).total_seconds()
    logging.info(f'Training of the model {model_name} for fold {fold} finished in {execution_time} seconds \n')
    return w, loss


def evaluate(model_name, w, x_val, y_val):
    """ Running the evaluation loop """
    logging.info(f'Evaluating the model: {model_name}')
    w = np.array(w).reshape(-1, 1)

    # Choosing the correct prediction function
    if 'logistic_regression' in model_name:
        pred_eval = np.round(sigmoid(x_val.dot(w)))
    else:
        pred_eval = np.round(x_val.dot(w))

    # Computing the accuracy from the given predictions
    if len(np.array(w).shape) == 1:
        accuracy_eval = np.mean(pred_eval == y_val) * 100
    else:
        accuracy_eval = np.mean(pred_eval == np.tile(y_val, (pred_eval.shape[1], 1)).T, axis=0) * 100

    logging.info(f'Accuracy of the {model_name} on the validation set: {accuracy_eval} \n')
    return accuracy_eval


def choose_hyperparams(cfg):
    """ Choose the hyperparameters and return them as:
        np.array([[initial_gamma, final_gamma, gamma_decay, _lambda, poly_degree]]) """
    if cfg.grid_search:
        logging.info(f'Running the hyperparameter Grid Search!')
        init_gamma = [0.01, 0.05, 0.1, 0.2]
        final_gamma = [0.001, 0.0005, 0.0001]
        gamma_decay = [0.3, 0.5, 0.7]
        _lambda = [0.05, 0.1, 0.2]
        poly_degree = [1, 2, 3, 4, 5]
        hyperparams = prepare_gs(init_gamma, final_gamma, gamma_decay, _lambda, poly_degree)
    else:
        logging.info(f'Default hyperparameters are used for the training!\n')
        init_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 0.1
        poly_degree = 2 if cfg.polynomial_features else 0
        hyperparams = np.array([[init_gamma, final_gamma, gamma_decay, _lambda, poly_degree]])
    return hyperparams


def choose_best_result(cfg, hyperparams, hyperparam_acc, hyperparam_pred, model, loss, test_data):
    """ Choose the model and corresponding hyperparams with the best performance """
    max_idx = np.argmax(hyperparam_acc)
    final_test_pred = hyperparam_pred[max_idx]
    final_test_pred[final_test_pred == 0] = -1
    init_gamma, fin_gamma, gamma_dec, _lambda, poly_degree = hyperparams[max_idx, :]
    acc = round(hyperparam_acc[max_idx], 2)

    # store the results in the .yaml file
    results = {"model": model,
               "accuracy": acc.item(),
               "epochs": cfg.n_epochs,
               "loss": loss.item(),
               "hyperparams": {
                   "initial_gamma": init_gamma.item(),
                   "final_gamma": fin_gamma.item(),
                   "gamma_decay": gamma_dec.item(),
                   "_lambda": _lambda.item(),
                   "poly_degree": poly_degree.item()}
               }
    model_config = Config(default_cfg_path=None, **results)
    model_config_path = os.path.join(cfg.work_dir, f"{model}_best_result.yaml")
    model_config.write_cfg(model_config_path)

    # create a csv submission for the AICrowd.com
    csv_submission_path = os.path.join(cfg.work_dir, f"{model}_submission.csv")
    create_submission(csv_submission_path, test_data['id_test'], final_test_pred)

    logging.info(f'===================================================================================================')
    logging.info(f'{model} -- Average accuracy after cross validation: {acc}')
    logging.info(f'The best set of hyperparameters for {model}: ')
    logging.info(f'init_gamma: {init_gamma}; fin_gamma: {fin_gamma}; gamma_decay: {gamma_dec}; _lambda: {_lambda};')
    logging.info(f'Results stored in: {model_config_path}')
    logging.info(f'Submission generated in: {csv_submission_path}')
    logging.info(f'===================================================================================================')


def choose_model(model_name, y_train, x_train, lambda_, w_initial, max_iters, gamma):
    """ Choose and train the model based on the given model_name """

    if model_name == 'mean_squared_error_gd':
        # the best degree for mean_squared_error_gd=2
        w, loss = mean_squared_error_gd(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'mean_squared_error_sgd':
        # the best degree for mean_squared_error_sgd=1, in this case we are adding the bias term using a batch_size
        # greater than 1 (batch_size=1 as a requirement) we should use degree 2
        w, loss = mean_squared_error_sgd(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'least_squares':
        # the best degree for least_squares=1, in this case we are adding the bias term
        w, loss = least_squares(y_train, x_train)

    elif model_name == 'ridge_regression':
        # the best experimental degree for ridge_regression=3
        w, loss = ridge_regression(y_train, x_train, lambda_)

    elif model_name == 'logistic_regression':
        # the best experimental degree for logistic_regression=3
        w, loss = logistic_regression(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'reg_logistic_regression':
        # the best experimental degree for reg_logistic_regression=3
        w, loss = reg_logistic_regression(y_train, x_train, lambda_, w_initial, max_iters, gamma)

    elif model_name == 'logistic_regression_bfgs':
        # the best experimental degree for logistic_regression=3
        w, loss = logistic_regression_bfgs(y_train, x_train, w_initial, max_iters, gamma)

    elif model_name == 'reg_logistic_regression_bfgs':
        # the best experimental degree for reg_logistic_regression=3
        w, loss = reg_logistic_regression_bfgs(y_train, x_train, lambda_, w_initial, max_iters, gamma)

    return w, loss
