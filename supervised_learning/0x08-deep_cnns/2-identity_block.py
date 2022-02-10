#!/usr/bin/env python3
"""
This module defines a function that builds
an identity block.
"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer.
    filters contains the filters for each convolution.
    """
    F11, F3, F12 = filters
    relu = K.activations.relu
    he_norm = K.initializers.he_normal()
    Conv2D = K.layers.Conv2D
    BatchNorm = K.layers.BatchNormalization

    Conv11 = Conv2D(filters=F11,
                    kernel_initializer=he_norm,
                    padding='same',
                    kernel_size=(1, 1))(A_prev)
    Batch11 = BatchNorm(axis=3)(Conv11)
    Relu11 = K.layers.Activation(relu)(Batch11)

    Conv3 = Conv2D(filters=F3,
                   kernel_initializer=he_norm,
                   padding='same',
                   kernel_size=(3, 3))(Relu11)
    Batch3 = BatchNorm(axis=3)(Conv3)
    Relu3 = K.layers.Activation(relu)(Batch3)

    Conv12 = Conv2D(filters=F12,
                    kernel_initializer=he_norm,
                    padding='same',
                    kernel_size=(1, 1))(Relu3)
    Batch12 = BatchNorm(axis=3)(Conv12)
    add = K.layers.Add()([Batch12, A_prev])
    return K.layers.Activation(relu)(add)
