""" Main logic for the training """

import logging
from datetime import datetime


def run_training(cfg, train_data):
    """ Running the training """
    logging.info(f'Starting the training!')

    # schedule training for each of the selected models
    for model, to_train in cfg.model_selection.items():
        if to_train:
            train(cfg, model, train_data)


def train(cfg, model_name, train_data):
    """ Running the training """
    start_time = datetime.now().replace(microsecond=0)
    logging.info(f'Training the model: {model_name} - started training at {start_time} for {cfg.n_epochs} epochs\n')
    raise NotImplementedError("Training is not implemented yet!")
