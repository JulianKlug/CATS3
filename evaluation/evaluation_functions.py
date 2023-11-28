import numpy as np


def root_mean_square_error(x1, x2):
    """
    Calculate the root mean square error between two vectors (x1 and x2)
    :param x1:
    :param x2:
    :return: rmse
    """
    if len(x1) != len(x2):
        raise ValueError(f'x1 and x2 should be of same length and not {len(x1)} != {len(x2)}')
    rmse = np.sqrt(((x1 - x2) ** 2).mean())
    return rmse