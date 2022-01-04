#!/usr/bin/env python3
"""This module defines a class called Neural Network"""

import numpy as np


class NeuralNetwork:
    """NeuralNetwork class that performs binary classification."""

    def __init__(self, nx, nodes):
        """
        Init method:
        nx is the number of input features
        nodes is the number of nodes in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
