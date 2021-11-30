#!/usr/bin/env python3
import numpy as np

""" This module has a function that adds two arrays. """


def add_arrays(arr1, arr2):
    ar1 = np.array(arr1)
    ar2 = np.array(arr2)
    new_list = []
    if ar1.shape != ar2.shape:
        return None
    new_list = np.add(ar1, ar2).tolist()
    return new_list
