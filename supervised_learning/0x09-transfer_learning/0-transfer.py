"""
This script defines preprocess_data and main.
preprocess_data - takes in input data and returns
the preprocessed data.
main - trains a model to classify the CIFAR 10 dataset.
"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    X - is the input data.
    Y - is the labels for the input data.
    Preprocesses data for the model.
    Returns X_p and Y_p respectively.
    """
    preprocess = K.applications.mobilenet.preprocess_input
    X_p = preprocess(X, data_format="channels_last")
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


def main():
    """
    Trains a model, compiles it and saves it as "cifar10.h5"
    """
    pass
