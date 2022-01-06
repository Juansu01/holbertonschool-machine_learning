#!/usr/bin/env python3
"""This module defines a function that returns
the accuracy of a prediction. """

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates accuracy"""
    y_pred = tf.math.argmax(y_pred, axis=1)
    y = tf.math.argmax(y, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.math.equal(y_pred, y), "float"))
    return acc
