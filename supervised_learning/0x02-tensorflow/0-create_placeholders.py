#!/usr/bin/env python3
"""This module defines a function that
creates placeholders. """

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Takes in the number of feature columns, and clasess
    returns both place holders for nx and classes
    respectively."""
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, classes], name='y')

    return x, y
