#!/usr/bin/env python3
"""
This module defines a function that uses a neural network to make
a prediction.
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Network is the network that is going to be
    used for the prediction, and data will be the input
    for the network. Returns the prediction"""

    return network.predict(x=data, verbose=verbose)
