#!/usr/bin/env python3
"""
This module defines a function that builds a modified
 version of the LeNet-5 architecture using keras.
"""


import tensorflow.keras as K


def lenet5(X):
    """X: Contains the input images for the network.
    Returns a keras model compiled to use adam optimization
    and accuracy metrics.
    """

    weights = K.initializers.he_normal()
    act_relu = K.layers.Activation("relu")
    L1 = K.layers.Conv2D(kernel_size=(5, 5),
                         filters=6, padding="valid",
                         activation=act_relu,
                         kernel_initializer=weights)
    out1 = L1(X)

    L2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    out2 = L2(out1)

    L3 = K.layers.Conv2D(padding='valid', filters=16,
                         kernel_size=(5, 5),
                         activation=act_relu,
                         kernel_initializer=weights)
    out3 = L3(out2)

    L4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    out4 = L4(out3)
    flat_out = K.layers.Flatten()(out4)

    L5 = K.layers.Dense(120, activation=act_relu,
                        kernel_initializer=weights)
    out5 = L5(flat_out)

    L6 = K.layers.Dense(84, activation=act_relu,
                        kernel_initializer=weights)
    out6 = L6(out5)

    L7 = K.layers.Dense(10, activation=act_relu,
                        kernel_initializer=weights)
    out7 = L7(out6)

    softmax = K.layers.Softmax()(out7)
    model = K.Model(X, softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
