""" General purpose utility functions to limit duplication of code """

import csv
import logging
import os


def makepath(desired_path, isfile=False):
    """
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :param isfile: boolean value that indicates if the path already exists
    """
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):
            os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path):
            os.makedirs(desired_path)
    return desired_path


def makelogger(log_dir=None, mode="w"):
    """Initializes and configures the logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # if the logging directory is given, logs will be stored in a file
    if log_dir:
        makepath(log_dir, isfile=True)
        fh = logging.FileHandler("%s" % log_dir, mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def create_submission(submission_path, test_id, test_pred):
    """Creates an output .csv file for submission to AICrowd.com"""
    with open(submission_path, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for row_1, row_2 in zip(test_id, test_pred):
            writer.writerow({"Id": int(row_1), "Prediction": int(row_2)})
