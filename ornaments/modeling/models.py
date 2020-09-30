from tensorflow.keras.layers import Dense, Conv2D, Dropout, AveragePooling2D, Flatten
from tensorflow.keras import Model, Sequential

from .layers import Autocorrelation2D


class LatticeModel(Model):

    def __init__(self):
        super(LatticeModel, self).__init__()

        self.cor = Autocorrelation2D()

        self.conv_net = Sequential([
            Conv2D(20, 5, strides=1, padding='same', kernel_regularizer='l2', bias_regularizer='l2'),
            Conv2D(40, 5, strides=1, padding='same', kernel_regularizer='l2', bias_regularizer='l2'),
            Dropout(0.2),
            Conv2D(40, 5, strides=1, padding='same', kernel_regularizer='l2', bias_regularizer='l2'),
            Dropout(0.2),
            AveragePooling2D(2, strides=2, padding='same')
        ])

        self.flatten = Flatten()

        self.fc = Sequential([
            Dense(128, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'),
            Dropout(0.2),
            Dense(5, activation='softmax', kernel_regularizer='l2', bias_regularizer='l2')
        ])

    def build(self, input_shape):
        self.cor.build(input_shape)
        input_shape = self.cor.compute_output_shape(input_shape)
        self.conv_net.build(input_shape)
        input_shape = self.conv_net.compute_output_shape(input_shape)
        self.flatten.build(input_shape)
        input_shape = self.flatten.compute_output_shape(input_shape)
        self.fc.build(input_shape)
        self.built = True

    def call(self, x, **kwargs):
        x = self.cor(x)
        x = self.conv_net(x)
        x = self.flatten(x)
        return self.fc(x)

