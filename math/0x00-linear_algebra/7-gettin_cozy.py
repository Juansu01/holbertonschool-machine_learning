#!/usr/bin/env python3
import numpy as np

""" This module defines function that concatenates two matrices. """


def cat_matrices2D(mat1, mat2, axis=0):
    a = np.array(mat1)
    b = np.array(mat2)
    return np.concatenate((a, b), axis).tolist()
