""" Functions that apply initial processing of data """

import numpy as np


def standardize_data(x, mean_x=None, std_x=None):
    """ Standardize the data (compute mean and std only for the training) """

    # compute mean only for the training data
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x

    # compute std only for the training data
    if std_x is None:
        std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def normalize_data(data, mean_train=None, std_train=None):
    """ Normalize the data """
    data, mean_data, std_data_new = standardize_data(data, mean_train, std_train)
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    return data, mean_data, std_data_new
