#!/usr/bin/env python3
""" This module defines function that concatenates two arrays. """


def cat_arrays(arr1, arr2):
    """This function concatenates two arrays."""
    return [i for i in [*arr1, *arr2]]
