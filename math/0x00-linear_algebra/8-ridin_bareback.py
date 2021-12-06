#!/usr/bin/env python3
""" This module defines a function that performs a matrix multiplication. """


def mat_mul(mat1, mat2):
    """Returns the product of the two matrices"""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for m in range(0, len(mat1)):
        rows = []
        for i in range(0, len(mat2[0])):
            columns = 0
            for j in range(0, len(mat2)):
                columns += mat1[m][j] * mat2[j][i]
            rows.append(columns)
        result.append(rows)
    return result
