#!/usr/bin/env python3
"""
This module defines a function that builds
a dense block as described in Densely
Connected Convolutional Networks.
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer.
    nb_filters is an integer representing the number
    of filers in X.
    growth_rate is the growth rate for the dense block.
    layers is the number of layers in the dense block.
    Returns the concatenated output of each layer within
    the Dense Block.
    """
    Conv2D = K.layers.Conv2D
    relu = K.activations.relu
    BatchNorm = K.layers.BatchNormalization
    he_normal = K.initializers.he_normal()
    for layer in range(layers):
        Batch1 = BatchNorm(axis=3,)(X)
        Relu1 = K.layers.Activation(relu)(Batch1)
        C1 = Conv2D(filters=(growth_rate * 4),
                    padding='same',
                    kernel_initializer=he_normal,
                    kernel_size=(1, 1))(Relu1)
        Batch2 = BatchNorm(axis=3)(C1)
        Relu2 = K.layers.Activation(relu)(Batch2)
        C2 = Conv2D(filters=(growth_rate),
                    padding='same',
                    kernel_initializer=he_normal,
                    kernel_size=(3, 3))(Relu2)
        nb_filters += growth_rate
        X = K.layers.concatenate([X, C2])

    return X, nb_filters
