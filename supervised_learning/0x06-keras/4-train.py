#!/usr/bin/env python3
"""This module defines a function that trains a model and
returns the history."""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """Takes in the model, with data, labels, batch size,
    epochs, verbose, shuffle to train the model using
    mini-batch gradient descent. """

    return network.fit(data, labels, batch_size,
                       epochs, verbose, shuffle)
