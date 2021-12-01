#!/usr/bin/env python3
""" This module has a function that flips a matrix. """


def matrix_transpose(matrix):
    """ Returns a flipped matrix. """
    new_matrix = [list(i) for i in zip(*matrix)]
    return new_matrix
