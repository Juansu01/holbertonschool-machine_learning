#!/usr/bin/env python3
"""This module defines a function that returns
the accuracy of a prediction. """

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates accuracy"""
    m = tf.keras.metrics.Accuracy()
    m.update_state(
                    y_true=y,
                    y_pred=y_pred)
    accuracy = m.result()
    return tf.math.reduce_mean(accuracy)
