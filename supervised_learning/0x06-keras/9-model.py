#!/usr/bin/env python3
"""This module defines two functions, the first
function saves a model and the second one, loads
a model."""


import tensorflow.keras as K


def save_model(network, filename):
    """Saves a the model "network" and uses
    "filename" as a name. Returns none"""

    network.save(filename)
    return None


def load_model(filename):
    """Takes in "filename" to load a specific
    model with that name. Returns the model."""

    return K.models.load_model(filename)
