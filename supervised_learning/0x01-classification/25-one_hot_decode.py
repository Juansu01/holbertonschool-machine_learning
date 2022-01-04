#!/usr/bin/env python3
"""This module defines a function that
decodes a one-hot matrix."""

import numpy as np


def one_hot_decode(one_hot):
    """Returns numpy array with the numeric
    labels, returns none if it fails."""
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2:
        return None
    one_hot = one_hot.transpose()
    return one_hot.argmax(axis=1)
