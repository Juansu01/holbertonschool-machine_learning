#!/usr/bin/env python3
""" This module has a function that returns the shape of a matrix. """


def matrix_shape(matrix):
    """ This function iterates over a given matrix and returns its shape """
    shape = []
    shape.append(len(matrix))
    shape.append(len(matrix[0]))
    if len(matrix[0]) > 2:
        shape.append(len(matrix[0][0]))
    return shape
