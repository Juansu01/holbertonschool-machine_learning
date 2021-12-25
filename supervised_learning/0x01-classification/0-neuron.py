#!/usr/bin/env python3
import numpy as np
""" This module defines the Neuron class, that
defines a single neuron that performs binary classification.
"""


class Neuron():
    """
    This class defines a single neuron performing binary
    classification.
    """
    def __init__(self, nx):
        """Init method."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
