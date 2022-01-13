#!/usr/bin/env python3
"""This module defines a function that
shuffles two matrices."""

import numpy as np


def shuffle_data(X, Y):
    """Returns both shuffled matrices. """
    m = X.shape[0]
    pattern = np.random.permutation(m)
    return (X[pattern], Y[pattern])
