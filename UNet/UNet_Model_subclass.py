from tensorflow.keras import Model, layers
import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)


class UNet(Model):
    def __init__(self, num_classes=1):
        super().__init__()
        self.filters = [64, 128, 256, 512]
        self.num_classes = num_classes
        self.skip_cons = []

    def double_conv(self, inputs, filter_size, add_skip=True):
        """
        This functions consists of 2 Conv2d layers with 3x3 kernels

        :param add_skip:
        :param inputs: input batch to the conv
        :param filter_size: number of 3x3 filters
        :return: the transformed inputs
        """
        conv1 = layers.Conv2D(filters=filter_size, kernel_size=3, padding='same', activation='relu')(inputs)
        conv2 = layers.Conv2D(filters=filter_size, kernel_size=3, padding='same', activation='relu')(conv1)

        if add_skip and filter_size != 1024:
            self.skip_cons.append(conv2)

        return conv2

    def encoder_block(self, inputs, filters):
        """
        This functions consists of the Encoder block.
        The Encoder block is: inputs -> double_conv() -> MaxPool2d()

        :param inputs: the input batch
        :param filters: number of filters
        :return: the transformed output from the encoder
        """
        for filter in filters:
            inputs = self.double_conv(inputs, filter_size=filter)
            inputs = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(inputs)
        return inputs

    def decoder_block(self, inputs, filters):
        """
        Decoder block consisting of a 2x2 upconv followed by 2 3x3 convs
        :param inputs: the batch of inputs
        :param filters: number of filters
        :return: the output of the decoder block
        """
        upsample_layer = layers.UpSampling2D(size=(2, 2))
        for filter in filters:
            inputs = layers.Conv2D(filters=filter, kernel_size=3, padding='same', activation='relu')(
                upsample_layer(inputs))
            skp = self.skip_cons.pop()
            inputs = layers.concatenate([skp, inputs], axis=3)
            inputs = self.double_conv(inputs=inputs, filter_size=filter, add_skip=False)

        return inputs

    def call(self, inputs, **kwargs):
        x = self.encoder_block(inputs=inputs, filters=self.filters)
        bottleneck = self.double_conv(inputs=x, filter_size=1024)
        outputs = self.decoder_block(inputs=bottleneck, filters=reversed(self.filters))

        outputs = layers.Conv2D(filters=self.num_classes, kernel_size=1, padding='same', activation='sigmoid')(outputs)

        return outputs


if __name__ == "__main__":
    image_shape = (256, 256, 1)

    model = UNet(num_classes=1)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    X = np.random.rand(1, *image_shape)
    y = np.random.rand(1, *image_shape)

    model.fit(X, y)
