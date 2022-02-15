#!/usr/bin/env python3
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
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    input_layer = K.Input(shape=(32, 32, 3))
    scale_layer = K.layers.Lambda(
        lambda x: K.backend.resize_images(x,
                                          (224 // 32),
                                          (224 // 32),
                                          "channels_last")
    )(input_layer)

    MobileNet = K.applications.MobileNet(weights="imagenet",
                                         include_top=False,
                                         input_shape=(224, 224, 3))
    x = MobileNet(scale_layer, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(1024, activation='relu')(x)
    x = K.layers.Dense(1024, activation='relu')(x)
    x = K.layers.Dense(512, activation='relu')(x)
    preds = K.layers.Dense(10 ,activation='softmax')(x)

    model = K.Model(inputs=input_layer, outputs=preds)

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])

    history = model.fit(x=X_train, y=Y_train,
                        batch_size=32,
                        epochs=15,
                        validation_data=(X_test, Y_test),
                        verbose=True)

    model.save('cifar10.h5')

if __name__ == '__main__':
    """
    This script trains a model using the MobileNet keras
    application, the data set used for training will be the
    CIFAR 10 dataset.
    """
    main()
