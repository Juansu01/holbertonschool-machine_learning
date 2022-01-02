#!/usr/bin/env python3
""" This module defines the Neuron class, that
defines a single neuron that performs binary classification.
"""
from typing import Type
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
        """Calculates the cost of the model
        using logistic regression."""
        m = Y.shape[1]
        cost = -1/m * np.sum(
            np.multiply(
                np.log(A), Y) + np.multiply(np.log(1.0000001 - A), (1-Y)))
        return cost

    def evaluate(self, X, Y):
        """This method evaluates a neuron's predictions."""
        predictions = self.forward_prop(X)
        cost = self.cost(Y, predictions)
        filtered_predictions = np.where(predictions >= 0.5, 1, 0)
        return (filtered_predictions, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Iterates once using gradient descent."""
        m = Y.shape[1]
        der = A - Y
        new_weight = (1 / m) * (np.matmul(X, der.transpose()).transpose())
        new_bias = (1 / m) * (np.sum(der))
        self.__W = self.W - alpha * new_weight
        self.__b = self.b - alpha * new_bias

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """This method trains the neuron using the
        number of iterations, and alpha as the learning
        rate. """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0.0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            out = self.forward_prop(X)
            self.gradient_descent(X, Y, out)
        return self.evaluate(X, Y)
