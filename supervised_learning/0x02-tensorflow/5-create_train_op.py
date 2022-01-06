#!/usr/bin/env python3
"""This module creates a training operation for the
network."""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Returns an operation that trains
    the network using gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
