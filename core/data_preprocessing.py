""" Functions that apply initial processing of data """

import logging

import numpy as np


class Imputer:
    """
    Imputing the given values according to strategy

    Parameters:
        missing_values: The value to be imputed
        strategy: The strategy to be used for imputation
        fill_value: The value to be used for imputation, if strategy is constant
                    otherwise, we will compute it in fit method

    Methods:
        fit: fit the imputer with the given data
        transform: transform the data according to the fitted imputer
        fit_transform: fit and transform the data
    """

    def __init__(
        self, missing_values=np.nan, strategy="mean", fill_value=None, axis=None
    ):
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.axis = axis
        if strategy == "mean":
            self.strategy = self.mean_imputation
        elif strategy == "median":
            self.strategy = self.median_imputation
        elif strategy == "constant":
            self.strategy = lambda x: None
        else:
            logging.error("Invalid strategy for imputation")
            raise ValueError("Invalid strategy for imputation")

    def mean_imputation(self, x):
        """Calculate the mean"""
        if np.isnan(self.missing_values):
            self.fill_value = np.nanmean(x, axis=self.axis)
        else:
            self.fill_value = np.mean(x[x != self.missing_values], axis=self.axis)

    def median_imputation(self, x):
        """Calculate the median"""
        if np.isnan(self.missing_values):
            self.fill_value = np.nanmedian(x, axis=self.axis)
        else:
            self.fill_value = np.median(x[x != self.missing_values], axis=self.axis)

    def fit(self, x):
        self.strategy(x)

    def transform(self, x):
        if np.isnan(self.missing_values):
            if self.axis is None:
                x[np.isnan(x)] = self.fill_value
            elif self.axis == 0:
                for i in range(x.shape[1]):
                    x[np.isnan(x[:, i]), i] = self.fill_value[i]
            elif self.axis == 1:
                for i in range(x.shape[0]):
                    x[i, np.isnan(x[i, :])] = self.fill_value[i]
        else:
            if self.axis == None:
                x[x == self.missing_values] = self.fill_value
            elif self.axis == 0:
                for i in range(x.shape[1]):
                    x[x[:, i] == self.missing_values, i] = self.fill_value[i]
            elif self.axis == 1:
                for i in range(x.shape[0]):
                    x[i, x[i, :] == self.missing_values] = self.fill_value[i]
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def one_hot_encode(data, column):
    """One hot encode the given columnn of data and remove the encoded column"""
    unique_values = np.unique(data[:, column])
    for value in unique_values:
        data = np.concatenate((data, (data[:, column] == value).reshape(-1, 1)), axis=1)
    data = np.delete(data, column, axis=1)
    return data


def log_transform(data, columns):
    """Log transform the data"""
    pos_cols = np.all(data[:, columns] > 0, axis=0)
    data[:, columns[pos_cols]] = np.log(data[:, columns[pos_cols]])
    return data


def remove_outliers(data, columns):
    """Remove outliers from the data using the interquartile range method"""
    right, left = np.percentile(data[:, columns], [90, 10], axis=0)
    iqr = right - left
    min_value = left - 1.5 * iqr
    max_value = right + 1.5 * iqr
    mask = np.ones(data.shape[0], dtype=bool)
    for i in range(len(columns)):
        mask = (
            mask
            & (data[:, columns[i]] > min_value[i])
            & (data[:, columns[i]] < max_value[i])
        )
    return mask


def preprocess_data(cfg, train_data, test_data):
    """Data preprocessing, including feature engineering and normalization"""
    logging.info(f"Starting data preprocessing!")

    # replace -999 to nan
    nan_imputer = Imputer(missing_values=-999, strategy="constant", fill_value=np.nan)
    train_data["x_train"] = nan_imputer.fit_transform(train_data["x_train"])
    test_data["x_test"] = nan_imputer.transform(test_data["x_test"])

    # impute nan values
    mean_imputer = Imputer(missing_values=np.nan, strategy="mean", axis=0)
    train_data["x_train"] = mean_imputer.fit_transform(train_data["x_train"])
    test_data["x_test"] = mean_imputer.transform(test_data["x_test"])

    # remove outliers
    outlier_cols = np.array([0, 1, 2, 3, 5, 8, 10, 13, 16, 19, 21, 23, 26, 29])
    mask = remove_outliers(train_data["x_train"], outlier_cols)
    train_data["x_train"] = train_data["x_train"][mask]
    train_data["y_train"] = train_data["y_train"][mask]

    # log transform
    log_cols = np.array([0, 1, 2, 3, 5, 8, 10, 13, 16, 19, 21, 23, 26, 29])
    train_data["x_train"] = log_transform(train_data["x_train"], log_cols)
    test_data["x_test"] = log_transform(test_data["x_test"], log_cols)

    # one hot encode categorical features
    column = 22
    mask = train_data["x_train"][:, column] > 1
    train_data["x_train"][:, column][mask] = 2
    test_data["x_test"][:, column][test_data["x_test"][:, column] > 1] = 2
    train_data["x_train"] = one_hot_encode(train_data["x_train"], column)
    test_data["x_test"] = one_hot_encode(test_data["x_test"], column)

    # cols 0 to 28 are continous, 29 to 31 are categorical
    cont_columns = np.arange(29)

    # normalize train/test data
    train_data["x_train"][:, cont_columns], mean_train, std_train = normalize_data(
        train_data["x_train"][:, cont_columns]
    )
    test_data["x_test"][:, cont_columns], _, _ = normalize_data(
        test_data["x_test"][:, cont_columns], mean_train, std_train
    )

    # add bias term (skip when polynomial features are required because build_poly adds the bias term anyway)
    if not cfg.polynomial_features:
        train_data["x_train"] = add_bias(train_data["x_train"])
        test_data["x_test"] = add_bias(test_data["x_test"])

    logging.info(f"Data preprocessed successfully!")


def add_bias(data):
    """Add bias to the data"""
    return np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)


def normalize_data(data, mean_train=None, std_train=None):
    """Normalize the data"""
    data, mean_data, std_data_new = standardize_data(data, mean_train, std_train)
    return data, mean_data, std_data_new


def standardize_data(x, mean_x=None, std_x=None):
    """Standardize the data (compute mean and std only for the training)"""

    # compute mean only for the training data
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x

    # compute std only for the training data
    if std_x is None:
        std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x
