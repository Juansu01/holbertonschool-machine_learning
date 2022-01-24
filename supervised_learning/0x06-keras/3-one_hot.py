"""This module defines a function that converts labels
into one-hot encoding matrix."""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Takes in the list of labels, and the classes argument
    then uses """

    return K.utils.to_categorical(labels, classes)
