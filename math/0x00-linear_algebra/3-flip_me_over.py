#!/usr/bin/env python3
import numpy as np

""" This module has a function that flips a matrix. """


def matrix_transpose(matrix):
    arr = np.array(matrix)
    return arr.transpose().tolist()
