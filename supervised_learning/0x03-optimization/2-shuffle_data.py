#!/usr/bin/env python3
"""This module defines a function that
shuffles two matrices."""

import numpy as np


def shuffle_data(X, Y):
    """Returns both shuffled matrices. """
    X = np.random.permutation(X)
    Y = np.random.permutation(Y)
    return (X, Y)
