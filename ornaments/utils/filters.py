import numpy as np
import tensorflow as tf


def gaussian(x, mu=0, sigma=1):
    """Calculates density function of normal distribution element-wise."""
    return tf.math.exp(-(x - mu)**2 / (2*sigma**2)) / tf.math.sqrt(2*np.pi*sigma**2)


def gaussian2d(size, mu=0, sigma=1):
    """Makes 2D gaussian kernel for convolution."""
    x = tf.range(start=-size, limit=size + 1, dtype=tf.float32)
    vals = gaussian(x, mu, sigma)
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = tf.expand_dims(gauss_kernel, 2)
    gauss_kernel = tf.expand_dims(gauss_kernel, 3)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def gaussian_filter(image, sigma):
    """Apply a 2-dimensional gaussian filter to each channel in the input tensor.
    """
    size = int(4*sigma + 0.5)
    kernel = gaussian2d(size, sigma=sigma)
    image = tf.expand_dims(tf.transpose(image, [3, 0, 1, 2]), 4)
    image = tf.map_fn(lambda x: tf.nn.conv2d(x, kernel, strides=1, padding='SAME'), image)
    return tf.transpose(tf.squeeze(image, 4), [1, 2, 3, 0])


def maximum_filter(image, size=None):
    """Calculate a 2-dimensional maximum filter.
    """
    return tf.nn.max_pool2d(image, ksize=size, strides=1, padding='SAME')
