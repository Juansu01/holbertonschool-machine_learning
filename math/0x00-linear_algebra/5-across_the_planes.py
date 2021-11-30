#!/usr/bin/env python3
import numpy as np

""" This module has a function that adds two arrays. """


def add_matrices2D(mat1, mat2):
    matrix1 = np.array(mat1)
    matrix2 = np.array(mat2)
    new_matrix = []
    if matrix1.shape != matrix2.shape:
        return None
    new_matrix = np.add(matrix1, matrix2)
    return new_matrix.tolist()
