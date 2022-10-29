""" Main logic for the data preparation """

import os
import numpy as np
import logging

from core.data_preprocessing import preprocess_data


def prepare_data(cfg):
    """ Loads and preprocesses the data """
    logging.info(f'Data preparation has started!')

    # load the data
    train_data, test_data = dataloader(cfg.dataset_dir)

    # preprocess the data
    preprocess_data(cfg, train_data, test_data)
    return train_data, test_data


def dataloader(data_path):
    """ Loads training/test data from the given data path """
    logging.info(f'Loading dataset from: {data_path}')

    # paths to the train and test data files
    train_data_path = os.path.join(data_path, 'train.csv')
    test_data_path = os.path.join(data_path, 'test.csv')

    # load train/test data from the given directories
    id_train, y_train, x_train = load_data(train_data_path)
    id_test, y_test, x_test = load_data(test_data_path)

    # store data in the dictionaries
    train_data = {"id_train": id_train, "y_train": y_train, "x_train": x_train}
    test_data = {"id_test": id_test, "y_test": y_test, "x_test": x_test}
    logging.info(f'Successfully loaded train and test data!')
    return train_data, test_data


def load_data(split_path):
    """ Loads data from the actual .csv files """
    data = np.loadtxt(split_path, delimiter=',', skiprows=1, converters={1: lambda x: int(x == 's'.encode('utf-8'))})
    return data[:, 0], data[:, 1], data[:, 2:]
