#!/usr/bin/env python3
""" This module has a function that uses extended slicing."""


def np_slice(matrix, axes={}):
    """Slices an array and returns result."""
    slc = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        slc[key] = slice(*value)
    return matrix[tuple(slc)]
