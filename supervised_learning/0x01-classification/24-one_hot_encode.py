#!/usr/bin/env python3
"""Convert numeric label into
hot matrix."""

import numpy as np


def one_hot_encode(Y, classes):
    """Takes in an array and turns it into a
    hot matrix."""
    Y = np.array(Y)
    Z = np.zeros((Y.size, Y.max() + 1))
    Z[np.arange(Y.size), Y] = 1
    return Z.transpose()
