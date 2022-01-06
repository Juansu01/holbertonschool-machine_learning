#!/usr/bin/env python3
"""This module defines a function that returns
the softmax loss. """

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates loss"""
    loss = tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)
    return loss
