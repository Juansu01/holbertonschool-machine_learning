#!/usr/bin/env python3
"""Convert numeric label into
hot matrix."""

import numpy as np


def one_hot_encode(Y, classes):
    """Takes in an array and turns it into a
    hot matrix."""

    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        Z = np.eye(classes)[Y]
        return Z.transpose()
    except Exception:
        return None
