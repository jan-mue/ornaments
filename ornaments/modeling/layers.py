import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import image_ops
from tensorflow.python.keras.utils import conv_utils

from ..point_groups import t
from ..utils import autocorrelation, correlation, affine_to_projective, rotation, reflection


class Autocorrelation2D(Layer):

    def __init__(self, data_format=None, **kwargs):
        super(Autocorrelation2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_last':
            # force format channels first for FFT operators
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        outputs = autocorrelation(inputs, normalize=True)
        outputs = tf.reduce_mean(outputs, axis=1)

        if self.data_format == 'channels_first':
            outputs = tf.expand_dims(outputs, 1)
        else:
            outputs = tf.expand_dims(outputs, 3)

        return outputs

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            shape[3] = 1
        else:
            shape[1] = 1
        return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'data_format': self.data_format
        }
        base_config = super(Autocorrelation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _get_transforms(t0, t1, t2):
    lattice_transform = np.eye(3)
    lattice_transform[:2, 0] = t1
    lattice_transform[:2, 1] = t2
    lattice_transform[:2, 2] = t0

    def rot(n):
        return affine_to_projective(rotation(2 * np.pi / n))

    e = np.eye(3)
    r2 = rot(2)
    r3 = rot(3)
    r4 = rot(4)
    r6 = rot(6)

    f_p = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])

    f_c = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

    # reflection about 150 degree line
    f_l = reflection((0, 0), rotation(5*np.pi/6) @ (1, 0))

    # reflection about 60 degree line
    f_s = reflection((0, 0), rotation(np.pi/3) @ (1, 0))

    r2 = t(0.5, 0.5) @ r2 @ t(-0.5, -0.5)
    r3 = t(0.5, 0.5) @ r3 @ t(-0.5, -0.5)
    r4 = t(0.5, 0.5) @ r4 @ t(-0.5, -0.5)
    r6 = t(0.5, 0.5) @ r6 @ t(-0.5, -0.5)

    f_p = t(0, 0.5) @ f_p @ t(0, -0.5)
    f_l = t(0.5, 0.5) @ f_l @ t(-0.5, -0.5)
    f_s = t(0.5, 0.5) @ f_s @ t(-0.5, -0.5)

    transforms = [e, r2, r3, r4, r6, f_p, f_c, f_l, f_s]
    transforms = lattice_transform @ transforms @ np.linalg.inv(lattice_transform)
    transforms = tf.linalg.inv(tf.cast(transforms, tf.float32))
    transforms = image_ops.matrices_to_flat_transforms(transforms)
    return transforms


class Correlation2D(Layer):

    def __init__(self, data_format=None, transforms=None, **kwargs):
        super(Correlation2D, self).__init__(**kwargs)
        self.transforms = transforms
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        if self.transforms is None:
            self.transforms = _get_transforms((0, 0), (input_shape[1], 0), (0, input_shape[2]))
        self.transforms = tf.expand_dims(self.transforms, 1)
        super(Correlation2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        shape = tf.shape(inputs)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        nt = len(self.transforms)

        x = tf.tile(tf.expand_dims(inputs, 1), (1, nt, 1, 1, 1))
        y = tf.map_fn(lambda m: image_ops.image_projective_transform_v2(inputs, m, [h, w], 'NEAREST'),  self.transforms)
        y = tf.transpose(y, [1, 0, 2, 3, 4])

        x = tf.transpose(tf.reshape(x, [b*nt, h, w, c]), [0, 3, 1, 2])
        y = tf.transpose(tf.reshape(y, [b*nt, h, w, c]), [0, 3, 1, 2])

        outputs = correlation(x, y, normalize=True)
        outputs = tf.reduce_mean(outputs, axis=1)

        outputs = tf.reshape(outputs, [b, nt, h, w])

        if self.data_format == 'channels_last':
            outputs = tf.transpose(outputs, [0, 2, 3, 1])

        return outputs

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            shape[3] = len(self.transforms)
        else:
            shape[1] = len(self.transforms)
        return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'data_format': self.data_format
        }
        base_config = super(Correlation2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))