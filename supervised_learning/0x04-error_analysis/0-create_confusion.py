#!/usr/bin/env python3
"""This module defines a function that that
 creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Returns the matrix multiplication of the transposed
    labels and logits. """

    return np.matmul(labels.transpose(), logits)
