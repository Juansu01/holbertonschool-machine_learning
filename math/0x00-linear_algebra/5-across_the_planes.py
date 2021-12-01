#!/usr/bin/env python3
""" This module has a function that adds two arrays. """


def add_matrices2D(mat1, mat2):
    """This function adds two matrices element-wise"""
    new_matrix = [[], []]
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    for i, e in enumerate(mat1):
        for j, k in enumerate(mat1[i]):
            new_matrix[i].append(mat1[i][j] + mat2[i][j])
    return new_matrix
