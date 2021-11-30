#!/usr/bin/env python3
import numpy as np

""" This module defines function that concatenates two arrays. """


def cat_arrays(arr1, arr2):
    ar1 = np.array(arr1)
    ar2 = np.array(arr2)
    return np.concatenate([ar1, ar2]).tolist()
