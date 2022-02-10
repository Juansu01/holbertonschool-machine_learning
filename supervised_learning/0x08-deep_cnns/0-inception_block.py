#!/usr/bin/env python3
"""
This module defines a function that builds an inception
block.
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Takes in "A_prev" which is the output
    of the previous layer and "filters" which
    is an array containing the numbers of filters
    for each convolution."""
    relu = K.activations.relu
    F1, F3R, F3, F5R, F5, FPP = filters
    Conv2D = K.layers.Conv2D
    kernel_init = K.initializers.he_normal

    first_layer = Conv2D(filters=F1,
                         kernel_size=(1, 1),
                         padding='same',
                         activation=relu,
                         kernel_initializer=kernel_init())
    first_output = first_layer(A_prev)

    second_layer = Conv2D(filters=F3R,
                          kernel_size=(1, 1),
                          padding='same',
                          activation=relu,
                          kernel_initializer=kernel_init())
    second_output = second_layer(A_prev)

    third_layer = Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         activation=relu,
                         kernel_initializer=kernel_init())
    third_output = third_layer(second_output)

    fourth_layer = Conv2D(filters=F5R,
                          kernel_size=(1, 1),
                          padding='same',
                          activation=relu,
                          kernel_initializer=kernel_init())
    fourth_output = fourth_layer(A_prev)

    fifth_layer = Conv2D(filters=F5,
                         kernel_size=(5, 5),
                         padding='same',
                         activation=relu,
                         kernel_initializer=kernel_init())
    fifth_output = fifth_layer(fourth_output)

    sixth_layer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same')
    sixth_output = sixth_layer(A_prev)
    seventh_layer = Conv2D(filters=FPP,
                           kernel_size=(1, 1),
                           padding='same',
                           activation=relu,
                           kernel_initializer=kernel_init())
    seventh_output = seventh_layer(sixth_output)

    return K.layers.concatenate([first_output,
                                 third_output,
                                 fifth_output,
                                 seventh_output])
