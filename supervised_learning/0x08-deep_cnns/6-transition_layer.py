#!/usr/bin/env python3
"""
This module defines a function that builds a
transition layer.
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer.
    nb_filters is the number of filters in X
    compression is the compression factor for the layer.
    """

    he_norm = K.initializers.he_normal()
    relu = K.activations.relu
    Batch1 = K.layers.BatchNormalization(axis=3)(X)
    Relu = K.layers.Activation(relu)(Batch1)
    nb_filters *= compression
    Conv1 = K.layers.Conv2D(filters=int(nb_filters),
                            padding='same',
                            kernel_initializer=he_norm,
                            kernel_size=(1, 1))(Relu)
    AvgPol1 = K.layers.AveragePooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid")(Conv1)

    return AvgPol1, int(nb_filters)
