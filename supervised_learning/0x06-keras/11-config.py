#!/usr/bin/env python3
"""
This module defines two funtions, the first saves
a models config, and the second one loads the config.
"""


import tensorflow.keras as K


def save_config(network, filename):
    """Saves a networks configuration, takes in the network
    and the filename to save it. Doesn't return"""

    my_json = network.to_json()
    with open(filename, 'w+') as file:
        file.write(my_json)
    return None


def load_config(filename):
    """Loads a models configuration using its
    filename. """

    with open(filename, 'r') as file:
        config = file.read()

    return K.models.model_from_config(config)
