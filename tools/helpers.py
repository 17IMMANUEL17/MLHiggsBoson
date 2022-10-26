""" Helper functions for various purposes """

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x=np.array(x)
    poly = np.ones((x.shape[0], 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def batch_iter(y, tx, batch_size, num_batches, shuffle = True) :
    """Iterate through data in batches.
    Args:
        tx (np.ndarray): Features data
        y (np.ndarray): Labels data
        batch_size (int, optional): Batch size. Defaults to None (i.e. full batch)
        num_batches (int, optional): Number of batches to iterate through. Defaults to None (i.e. use all data)
        shuffle (bool, optional): Whether to shuffle the data before generating batches. Defaults to True.
    Yields:
        Generator: (tx_batch, y_batch)
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        y = y[shuffle_indices]
        tx = tx[shuffle_indices]

    batch_size = batch_size or len(tx)
    batches = int(np.ceil(len(tx) // batch_size))
    num_batches = num_batches or batches

    for i in range(num_batches):
        yield y[i*batch_size:(i+1)*batch_size], tx[i*batch_size:(i+1)*batch_size]


def kfold_cross_validation(data, folds_num=5):
    """ Divides all data samples into groups in order to use the whole dataset for training and validation """
    data_idx = np.arange(data.shape[0])
    test_idx, train_idx = [], []

    # size of each fold
    fold_size = int(data.shape[0] / folds_num)
    for i in range(folds_num):

        test_fold = data_idx[i * fold_size: (i + 1) * fold_size]
        train_fold = np.delete(data_idx, test_fold)

        test_idx.append(test_fold)
        train_idx.append(train_fold)

    return test_idx, train_idx

