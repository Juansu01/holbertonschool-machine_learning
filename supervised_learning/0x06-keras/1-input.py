#!/usr/bin/env python3
"""This script defines a function that
 builds a neural network with the Keras library."""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Takes in the number of input features, layers contains the
    number of nodes in a specific layer, activations contains a list of
    the activation functions, lambtha holds the L2 regularization parameter
    keep_prob is the probability of a node being kept for dropout."""

    reg = K.regularizers.l2(lambtha)
    layer_list = []
    inp = K.Input(shape=(nx,))
    x = inp

    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i],
                               activation=activations[i],
                               kernel_regularizer=reg)(x)
            continue
        x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=inp, outputs=x)
