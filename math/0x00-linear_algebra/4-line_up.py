#!/usr/bin/env python3
""" This module has a function that adds two arrays. """


def add_arrays(arr1, arr2):
    """Checks if both arrays have the same shape and adds them element-wise."""
    new_list = []
    if len(arr1) == len(arr2):
        for i, e in enumerate(arr1):
            new_list.append(arr1[i] + arr2[i])
        return new_list
    return None
