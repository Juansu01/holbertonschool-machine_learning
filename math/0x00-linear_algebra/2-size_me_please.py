#!/usr/bin/env python3
import numpy as np

""" This module has a function that returns the shape of a matrix. """


def matrix_shape(matrix):
    """ Returns the shape of a matrix. """
    arr = np.array(matrix)
#    shape = []
#    shape.append(len(matrix))
#    shape.append(len(matrix[0]))
#    if len(matrix[0]) > 2:
#        shape.append(len(matrix[0][0]))
    return list(arr.shape)
