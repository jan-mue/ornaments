import numpy as np
import tensorflow as tf
from tensorflow.python.ops import image_ops
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import get_rotation_matrix
from skimage.draw import polygon
from skimage.transform import resize

from .filters import maximum_filter
from .math import rotatedRectWithMaxArea


@tf.function
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: tensor of shape (B, H, W)
    - y: tensor of shape (B, H, W)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)

    Source: https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


@tf.function
def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.

    Source: https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


@tf.function
def rotate_with_crop(images, angle):

    shape = tf.shape(images)
    shape = tf.cast(shape, tf.float32)

    if len(shape) == 4:
        height = shape[1]
        width = shape[2]
    elif len(shape) == 3:
        images = tf.expand_dims(images, 0)
        height = shape[0]
        width = shape[1]
    else:
        raise ValueError('images must be tensor of rank 3 or 4.')

    r = get_rotation_matrix([angle], height, width)
    images = image_ops.image_projective_transform_v2(images, r, [height, width], 'BILINEAR')

    w_r, h_r = rotatedRectWithMaxArea(width, height, angle)
    w_offset = tf.cast(tf.round((width - w_r) / 2), tf.int32)
    h_offset = tf.cast(tf.round((height - h_r) / 2), tf.int32)
    w_r, h_r = tf.cast(tf.round(w_r), tf.int32), tf.cast(tf.round(h_r), tf.int32)

    images = tf.image.crop_to_bounding_box(images, h_offset, w_offset, h_r, w_r)

    if len(shape) == 3:
        return images[0]

    return images


def polygon_mask(vertices, shape):
    vertices = np.asarray(vertices)
    img = np.zeros(shape, dtype=bool)
    rr, cc = polygon(vertices[:, 1], vertices[:, 0])
    img[rr, cc] = True
    return img


def resize_with_pad(image, target_height, target_width):

    height, width, _ = image.shape

    # Find the ratio by which the image must be adjusted
    # to fit within the target
    ratio = max(width / target_width, height / target_height)
    resized_height_float = height / ratio
    resized_width_float = width / ratio
    resized_height = np.floor(resized_height_float)
    resized_width = np.floor(resized_width_float)

    padding_height = (target_height - resized_height_float) / 2
    padding_width = (target_width - resized_width_float) / 2
    padding_height = np.floor(padding_height)
    padding_width = np.floor(padding_width)
    p_height = max(0, padding_height)
    p_width = max(0, padding_width)

    # Resize first, then pad to meet requested dimensions
    resized = resize(image, (resized_height, resized_width))

    after_padding_width = target_width - p_width - resized_width
    after_padding_height = target_height - p_height - resized_height
    paddings = np.reshape([p_height, after_padding_height, p_width, after_padding_width, 0, 0], (3, 2))

    return np.pad(resized, paddings.astype(int))


def extract_peaks(images, min_distance=1, threshold=None):
    """Finds peaks in a batch of images as list of coordinates.
    """
    shape = tf.shape(images)
    batch_size = shape[0]

    window_size = 2 * min_distance + 1
    image_max = maximum_filter(images, window_size)[:, :, :, 0]

    if threshold is None:
        # use image minimum as threshold to exclude points in a constant region
        threshold = -maximum_filter(-images, window_size)[:, :, :, 0]

    images = images[:, :, :, 0]
    is_max = tf.logical_and(tf.equal(images, image_max), tf.greater(images, threshold))
    return [tf.cast(tf.where(is_max[i]), tf.int32) for i in range(batch_size)]


def filtered_peaks(images, min_distance=1, threshold=None, percentile=0.9):
    """Extracts peaks and filters them by region of dominance.
    """
    peaks = extract_peaks(images, min_distance, threshold)
    images = np.asarray(images)

    result = []
    for idx, p in enumerate(peaks):
        p = np.asarray(p)

        if len(p) < 9:
            result.append(p)
            continue

        values = images[idx, p[:, 0], p[:, 1], 0]
        sp = p[np.argsort(values)[::-1]]

        i, j = np.indices((len(p), len(p)))
        dist = np.sum((sp[i] - sp[j])**2, axis=-1)
        dist[np.tril_indices(len(p))] = np.ma.minimum_fill_value(dist)
        dom_radius = np.min(dist, axis=0)

        sorting = np.argsort(dom_radius)[::-1]
        sp = sp[sorting]
        dom_radius = dom_radius[sorting]

        dom_threshold = percentile * dom_radius[8]
        sp = sp[dom_radius > dom_threshold]

        result.append(sp)

    return result
