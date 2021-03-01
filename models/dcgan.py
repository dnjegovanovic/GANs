import tensorflow as tf
import numpy as np


def make_dcgan_generator(z_size=20, output_size=(28, 28, 1),
                         n_filters=128, n_blocks=2):
    """
    :param z_size: input in generator
    :param output_size:
    :param n_filters: number of filters
    :param n_blocks: number of conv blocks
    :return:  model
    """

    size_factor = 2 ** n_blocks
    hidden_size = (output_size[0] // size_factor, output_size[1] // size_factor)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(z_size,)),

        tf.keras.layers.Dense(units=n_filters * np.prod(hidden_size),
                              use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((hidden_size[0], hidden_size[1], n_filters)),

        tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(5, 5), strides=(1, 1),
                                        padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ])

    nf = n_filters

    for i in range(n_blocks):
        nf = nf // 2
        model.add(
            tf.keras.layers.Conv2DTranspose(filters=nf, kernel_size=(5, 5), strides=(2, 2),
                                            padding='same', use_bias=False)
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(filters=output_size[2], kernel_size=(5, 5), strides=(1, 1),
                                        padding='same', use_bias=False, activation='tanh')
    )

    return model


def make_dcgan_discriminator(input_size=(28, 28, 1), n_filters=64, n_blocks=2):
    """
    :param input_size: fage image from generator
    :param n_filters: numbre of conv filters
    :param n_blocks: number of conv blocks
    :return: model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
        tf.keras.layers.Conv2D(filters=n_filters, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ])

    nf = n_filters
    for i in range(n_blocks):
        nf = nf * 2
        model.add(
            tf.keras.layers.Conv2D(filters=nf, kernel_size=(5, 5), strides=(2, 2), padding='same')
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(
        tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    )
    model.add(
        tf.keras.layers.Reshape((1,))
    )

    return model
