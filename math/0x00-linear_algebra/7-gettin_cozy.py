#!/usr/bin/env python3
""" This module defines function that concatenates two matrices. """


def cat_matrices2D(mat1, mat2, axis=0):
    """Returns a new matrix, which is the result of concatenation."""
    if (len(mat1) == len(mat2)) and axis == 1:
        new_matrix = []
        for row1, row2 in zip(mat1, mat2):
            copy = row1.copy()
            copy.extend(row2.copy())
            new_matrix.append(copy)
        return new_matrix
    if (len(mat1[0]) == len(mat2[0])) and axis == 0:
        new_matrix = []
        for i, row in enumerate(mat1):
            new_matrix.append(row.copy())
        for i, row in enumerate(mat2):
            new_matrix.append(row.copy())
        return new_matrix
    else:
        return None
