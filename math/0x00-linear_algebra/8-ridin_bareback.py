#!/usr/bin/env python3
import numpy as np

""" This module defines a function that performs a matrix multiplication. """


def mat_mul(mat1, mat2):
    """Returns the product of the two matrices"""
    a = np.array(mat1)
    b = np.array(mat2)
    if len(mat1[0]) != len(mat2):
        return None
    return np.matmul(a, b).tolist()
