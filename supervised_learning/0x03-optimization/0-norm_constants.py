#!/usr/bin/env python3
"""This module defines a function that calculates
 the normalization (standardization) constants of a matrix."""

import numpy as np


def normalization_constants(X):
    """Returns: the mean and standard
    deviation of each feature, respectively."""
    return (X.mean(axis=0), X.std(axis=0))
