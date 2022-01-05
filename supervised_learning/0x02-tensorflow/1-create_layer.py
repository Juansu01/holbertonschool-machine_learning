#!/usr/bin/env python3
"""This module defines a function that creates layers."""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Returns the new layer."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    new_layer = \
        tf.layers.Dense(
            n, activation=activation, name="layer",
            kernel_initializer=initializer)
    return new_layer(prev)
