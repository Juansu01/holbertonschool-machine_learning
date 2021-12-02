#!/usr/bin/env python3
""" This module defines function that concatenates two matrices. """


def cat_matrices2D(mat1, mat2, axis=0):
    """Returns a new matrix, which is the result of concatenation."""
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for i, e in enumerate(mat1[:]):
            new_matrix.append(e.copy())
        for i, e in enumerate(mat2[:]):
            new_matrix[i].append(e.copy()[0])
        return new_matrix
    if axis == 0:
        if len(mat1[0]) != len(mat1[0]):
            return None
        return mat1 + mat2
