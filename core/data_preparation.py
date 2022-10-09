""" Main logic for the data preparation """

import os
import numpy as np
import logging


def prepare_data(cfg):
    """ Loads and preprocesses the data """
    logging.info(f'Data preparation has started!')
    data = dataloader(cfg.dataset_dir)


def dataloader(data_path):
    """ Loads training/test data from the given data path """
    logging.info(f'Loading dataset from: {data_path}!')

    # data paths
    train_data_path = os.path.join(data_path, 'train.csv')
    test_data_path = os.path.join(data_path, 'test.csv')

    # load train/test data from the given directories
    id_train, y_train, x_train = load_data(train_data_path)
    id_test, y_test, x_test = load_data(test_data_path)
    data = np.concatenate((x_train, x_test), axis=0)
    logging.info(f'Successfully loaded train and test data!')
    return data


def load_data(split_path):
    """ Loads data from the actual .csv files """
    data = np.loadtxt(split_path, delimiter=',', skiprows=1, converters={1: lambda x: int(x == 's'.encode('utf-8'))})
    return data[:, 0], data[:, 1], data[:, 2:]
