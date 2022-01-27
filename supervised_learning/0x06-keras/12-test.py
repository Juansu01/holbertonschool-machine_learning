#!/usr/bin/env python3
"""
This module defines a function that evaluates a model.
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """This method tests a neural network and
    returns its loss and accuracy respectively."""

    loss, acc = network.evaluate(x=data, y=labels,
                                 verbose=verbose)

    return loss, acc
