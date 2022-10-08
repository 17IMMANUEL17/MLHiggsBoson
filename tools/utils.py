""" General purpose utility functions to limit duplication of code """

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
