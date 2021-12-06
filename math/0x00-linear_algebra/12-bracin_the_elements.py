#!/usr/bin/env python3
"""This module defines a function performs
element-wise addition, subtraction, multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """Returns the results of each matrix"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2,)
