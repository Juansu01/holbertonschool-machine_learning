#!/usr/bin/env python3
""" This module defines the Neuron class, that
defines a single neuron that performs binary classification.
"""
import numpy as np


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

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Privatizes the W attribute"""
        return self.__W

    @property
    def b(self):
        """Privatizes the b attribute"""
        return self.__b

    @property
    def A(self):
        """Privatizes the A attribute"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of a neuron."""
        x = np.matmul(self.W, X) + self.b
        self.__A = 1/(1 + np.exp(-x))
        return (self.A)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m =  Y.shape[1]
        cost = -1/m * np.sum( np.multiply(np.log(A), Y) + np.multiply(np.log(1.0000001 - A), (1-Y)))
