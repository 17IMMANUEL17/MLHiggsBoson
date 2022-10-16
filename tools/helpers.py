""" Helper functions for various purposes """

import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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
