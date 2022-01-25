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

    for i in range(len(layers)):
        if i == 0:
            layer_list.append(K.layers.Dense(layers[i],
                                             activation=activations[i],
                                             kernel_regularizer=reg,
                                             input_shape=(nx,)))
            continue

        layer_list.append(K.layers.Dropout(1 - keep_prob))
        layer_list.append(K.layers.Dense(layers[i], activation=activations[i],
                                         kernel_regularizer=reg))
    return K.Sequential(layer_list)
