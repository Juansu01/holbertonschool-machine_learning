#!/usr/bin/env python3
"""
This module defines a function that builds an
inception network.
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    This function builds an inception network.
    input data has a shape of (224, 224, 3)
    all convolutions use ReLU activation
    Returns the keras model.
    """
    he_normal = K.initializers.he_normal()
    relu = K.activations.relu
    img_input = K.Input(shape=(224, 224, 3))
    Conv2D = K.layers.Conv2D
    MaxPol2D = K.layers.MaxPooling2D

    conv2 = Conv2D()
