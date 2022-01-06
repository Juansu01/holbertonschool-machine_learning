#!/usr/bin/env python3
"""This module defines a function that performs
forward propagation. """


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates layers using foward propagation."""
    create_layer = __import__('1-create_layer').create_layer
    for size, activation in zip(layer_sizes, activations):
        if size == layer_sizes[0]:
            pred = create_layer(x, size, activation)
            continue
        pred = create_layer(pred, size, activation)
    return pred
