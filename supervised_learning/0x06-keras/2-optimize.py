#!/usr/bin/env python3
"""This module defines a function that adds the Adam
    optimizer to a model."""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Network is the model to add the adam optimizer
    to, alpha is the learning rate, beta1 and beta2 are
    the first and second optimizer arguments, respectively."""

    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1, beta_2=beta2)
    network.compile(optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
