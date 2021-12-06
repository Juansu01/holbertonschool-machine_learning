#!/usr/bin/env python3
""" This module has a function that returns the shape of a matrix. """


def matrix_shape(matrix):
    """ This function iterates over a given matrix and returns its shape """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
