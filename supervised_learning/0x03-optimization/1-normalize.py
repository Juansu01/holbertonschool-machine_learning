#!/usr/bin/env python3
"""This module defines a function that
normalizes (standardizes) a matrix"""

import numpy as np


def normalize(X, m, s):
    """Returns: The normalized X matrix."""
    return (X - m) / s
