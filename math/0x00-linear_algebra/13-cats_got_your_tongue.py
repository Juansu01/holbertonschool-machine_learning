#!/usr/bin/env python3
"""This module defines a function that concatenates two matrices."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Returns a new list with concatenated matrices"""
    return np.concatenate((mat1, mat2), axis)
