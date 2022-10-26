""" Main logic for the training """

import logging
import numpy as np
from datetime import datetime
import os

from core.implementations import least_squares_GD, logistic_regression, reg_logistic_regression, ridge_regression, \
    least_squares_SGD, least_squares, hyperparams_gs
from core.costs import sigmoid
from tools.helpers import kfold_cross_validation 
from tools.cfg_parser import Config
from tools.utils import create_submission
from core.data_preprocessing import build_poly


def run_training(cfg, train_data, test_data, num_folds=5):
    """ Running the training """
    logging.info(f'Starting the training!')

    # choose the hyperparameters
    #hyperparams = choose_hyperparams(cfg,train_data)

    # schedule training for each of the selected models
    for model, to_train in cfg.model_selection.items():
        if to_train:
            hyperparams = choose_hyperparams(model,cfg,train_data)


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

                    # run the actual training loop
                    losses, ws, pol_x_val = train(cfg, fold, hyperparams[i, :], model, x_train, y_train,x_val)

                    # run evaluation on the val data
                    accuracy_eval = evaluate(model, ws, pol_x_val, y_val)
                    acc.append(accuracy_eval)
        
                    # run evaluation on the test data
                    if model == 'logistic_regression' or model == 'reg_logistic_regression':
                        pred_test = np.round(sigmoid(pol_x_val.dot(ws.T)))
                    else:
                        pred_test = np.round(pol_x_val.dot(ws))
                    pred.append(pred_test)

                # store the results to get the best hyperparameters
                hyperparam_acc.append(np.mean(acc))
                hyperparam_pred.append(np.round(np.mean(np.array(pred), axis=0)))

            # select the best model (the best hyperparam setting) and create the .csv submission
            choose_best_result(cfg, hyperparams, hyperparam_acc, hyperparam_pred, model, losses, test_data)


def train(cfg, fold, trial_hyperparams, model_name, x_train, y_train, x_val):
    """ Running the training loop """
    init_gamma, fin_gamma, gamma_dec, _lambda = trial_hyperparams
    logging.info(f'init_gamma: {init_gamma}; fin_gamma: {fin_gamma}; gamma_decay: {gamma_dec}; _lambda: {_lambda};')
    start_time = datetime.now().replace(microsecond=0)
    logging.info(f'Training the model: {model_name} for fold {fold} - started training at {start_time}')

    # select and train the model based on the given model_name
    losses, ws, pol_x_val = choose_model(model_name, y_train, x_train, _lambda, cfg.n_epochs, init_gamma, x_val)

    end_time = datetime.now().replace(microsecond=0)
    execution_time = (end_time - start_time).total_seconds()
    logging.info(f'Training of the model {model_name} for fold {fold} finished in {execution_time} seconds \n')
    return losses, ws, pol_x_val


def evaluate(model_name, ws, x_val, y_val):
    """ Running the evaluation loop """
    logging.info(f'Evaluating the model: {model_name}')
    ws = np.array(ws)
    #choosing the correct prediction function 
    if model_name == 'logistic_regression' or model_name == 'reg_logistic_regression':
        pred_eval = np.round(sigmoid(x_val.dot(ws.T)))
    else:
        pred_eval = np.round(x_val.dot(ws))
    if len(np.array(ws).shape) == 1:
        accuracy_eval = np.mean(pred_eval == y_val) * 100
    else:
        accuracy_eval = np.mean(pred_eval == np.tile(y_val, (pred_eval.shape[1], 1)).T, axis=0) * 100
    logging.info(f'Accuracy of the {model_name} on the validation set: {accuracy_eval} \n')
    return accuracy_eval


def choose_hyperparams(model, cfg, train_data):
    """ Choose the hyperparameters and return them as:
        np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]]) """
    if cfg.grid_search:
        logging.info(f'Running the hyperparameter Grid Search!')
        hyperparams= np.zeros((1,4))
        hyperparams= hyperparams_gs(model,cfg, train_data)
    else:
        logging.info(f'Default hyperparameters are used for the training!\n')
        initial_gamma = 0.01
        final_gamma = 0.0001
        gamma_decay = 0.5
        _lambda = 1e-4
        hyperparams = np.array([[initial_gamma, final_gamma, gamma_decay, _lambda]])
    return hyperparams


def choose_best_result(cfg, hyperparams, hyperparam_acc, hyperparam_pred, model, losses, test_data):
    """ Choose the model and corresponding hyperparams with the best performance """
    max_idx = np.argmax(hyperparam_acc)
    final_test_pred = hyperparam_pred[max_idx]
    final_test_pred[final_test_pred == 0] = -1
    init_gamma, fin_gamma, gamma_dec, _lambda = hyperparams[max_idx, :]
    acc = round(hyperparam_acc[max_idx], 2)

    # store the results in the .yaml file
    results = {"model": model,
               "accuracy": acc.item(),
               "epochs": cfg.n_epochs,
               "loss": losses.item(),
               "hyperparams": {
                   "initial_gamma": init_gamma.item(),
                   "final_gamma": fin_gamma.item(),
                   "gamma_decay": gamma_dec.item(),
                   "_lambda": _lambda.item()}
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


def choose_model(model_name, y_train, x_train, lambda_, max_iters, gamma, x_val):
    """ Choose and train the model based on the given model_name """

    # TODO: Unify if the models should return losses and ws as lists or just final loss and w
    if model_name == 'least_squares_GD':
        # build polynomial features: best degree for least_squares_GD=2
        pol_x_train = build_poly(x_train, 2)
        pol_x_val = build_poly(x_val, 2)

        # initialize the weights
        w_initial = np.zeros((pol_x_train.shape[1]))

        losses, ws = least_squares_GD(y_train, pol_x_train, w_initial, max_iters, gamma)
        w = ws[-1]
        loss = losses[-1]

    elif model_name == 'least_squares_SGD':
        # build polynomial features: best experimental degree for least_squares_SGD=1, in this case we are adding the bias term\
        #using a batch_size greater than 1 (batch_size=1 as a requirement) we should use degree 2
        pol_x_train = build_poly(x_train, 1)
        pol_x_val = build_poly(x_val, 1)

        # initialize the weights
        w_initial = np.zeros((pol_x_train.shape[1]))

        losses, ws = least_squares_SGD(y_train, pol_x_train, w_initial, max_iters, gamma)
        w = ws[-1]
        loss = losses[-1]

    elif model_name == 'least_squares':
        # build polynomial features: best experimental degree for least_squares=1, in this case we are adding the bias term
        pol_x_train = build_poly(x_train, 1)
        pol_x_val = build_poly(x_val, 1)

        # initialize the weights
        w_initial = np.zeros((pol_x_train.shape[1]))

        loss, w = least_squares(y_train, pol_x_train)

    elif model_name == 'ridge_regression':
        # build polynomial features: best experimental degree for ridge_regression=3
        pol_x_train = build_poly(x_train, 3)
        pol_x_val = build_poly(x_val, 3)

        # initialize the weights
        w_initial = np.zeros((pol_x_train.shape[1]))

        loss, w = ridge_regression(y_train, pol_x_train, lambda_)

    elif model_name == 'logistic_regression':
        # build polynomial features: best experimental degree for logistic_regression=3
        pol_x_train = build_poly(x_train, 3)
        pol_x_val = build_poly(x_val, 3)

        # initialize the weights
        w_initial = np.zeros((pol_x_train.shape[1]))

        losses, ws = logistic_regression(y_train, pol_x_train, w_initial, max_iters, gamma)
        w = ws[-1]
        loss = losses[-1]

    elif model_name == 'reg_logistic_regression':
        # build polynomial features: best experimental degree for reg_logistic_regression=3
        pol_x_train = build_poly(x_train, 3)
        pol_x_val = build_poly(x_val, 3)

        # initialize the weights
        w_initial = np.zeros((pol_x_train.shape[1]))

        losses, ws = reg_logistic_regression(y_train, pol_x_train, lambda_, w_initial, max_iters, gamma)
        w = ws[-1]
        loss = losses[-1]

    return loss, w, pol_x_val
