import os
import warnings

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

skip_cons = []


def double_conv(inputs, filter_size, add_skip=True):
    """
    This functions consists of 2 Conv2d layers with 3x3 kernels

    :param add_skip:
    :param inputs: input batch to the conv
    :param filter_size: number of 3x3 filters
    :return: the transformed inputs
    """
    conv1 = layers.Conv2D(filters=filter_size, kernel_size=3,
                          padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(filters=filter_size, kernel_size=3,
                          padding='same', activation='relu')(conv1)

    if add_skip and filter_size != 1024:
        skip_cons.append(conv2)

    return conv2


def encoder_block(inputs, filters):
    """
    This functions consists of the Encoder block.
    The Encoder block is: inputs -> double_conv() -> MaxPool2d()

    :param inputs: the input batch
    :param filters: number of filters
    :return: the transformed output from the encoder
    """
    for filter in filters:
        inputs = double_conv(inputs, filter_size=filter)
        inputs = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(inputs)
    return inputs


def decoder_block(inputs, filters):
    """
    Decoder block consisting of a 2x2 upconv followed by 2 3x3 convs
    :param inputs: the batch of inputs
    :param filters: number of filters
    :return: the output of the decoder block
    """
    for filter in filters:
        inputs = layers.Conv2DTranspose(
            filters=filter, kernel_size=2, padding='same', activation='relu', strides=(2, 2))(inputs)
        skp = skip_cons.pop()
        inputs = layers.concatenate([skp, inputs], axis=3)
        inputs = double_conv(inputs=inputs, filter_size=filter, add_skip=False)

    return inputs


def unet(inputs_shape=(256, 256, 1), num_classes=1):
    all_filters = [64, 128, 256, 512]
    inputs = layers.Input(shape=inputs_shape)
    x = encoder_block(inputs=inputs, filters=all_filters)
    bottleneck = double_conv(inputs=x, filter_size=1024)
    outputs = decoder_block(inputs=bottleneck, filters=reversed(all_filters))

    outputs = layers.Conv2D(filters=num_classes, kernel_size=1,
                            padding='same', activation='softmax')(outputs)

    model = keras.Model(inputs, outputs)

    return model


if __name__ == "__main__":

    image_shape = (256, 256, 1)

    model = unet(inputs_shape=image_shape)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    X = np.random.rand(1, *image_shape)
    y = np.random.rand(1, *image_shape)

    model.fit(X, y)

    model.summary()
