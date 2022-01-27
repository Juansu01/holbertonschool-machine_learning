#!/usr/bin/env python3
"""This module updates the train model function to also use early
stopping."""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Takes in the model, with data, labels, batch size,
    epochs, verbose, shuffle to train the model using
    mini-batch gradient descent. """

    my_callbacks = None
    if early_stopping and validation_data:
        my_callbacks = []
        my_callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='loss', patience=patience))

    return network.fit(x=data, y=labels,
                       batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       validation_data=validation_data,
                       callbacks=my_callbacks,
                       shuffle=shuffle)
