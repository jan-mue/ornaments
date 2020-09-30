import numpy as np
import tensorflow as tf
from geometer import Point, Transformation, Triangle, Circle, PointCollection


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def affine_to_projective(A=None, b=0):
    result = np.eye(3)
    if A is not None:
        result[:2, :2] = A
    result[:2, 2] = b
    return result


def translation(x, y):
    return affine_to_projective(b=(x, y))


def reflection(a, b):
    # reflection at line through a and b
    lx, ly = b[0] - a[0], b[1] - a[1]
    refl = np.array([[lx**2 - ly**2, 2*lx*ly, 0],
                     [2*lx*ly, ly**2 - lx**2, 0],
                     [0, 0, lx**2 + ly**2]])

    t = translation(*a)
    t_inv = translation(-a[0], -a[1])

    return t @ refl @ t_inv


def random_choice(seq):
    ind = tf.random.uniform([], 0, len(seq), dtype=tf.int32)
    return tf.gather(seq, ind)


def correlation(x, y, normalize=False):
    shape = tf.shape(x)
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]

    padding_shape_h = [b, c, h - 1, w]
    padding_shape_w = [b, c, 2*h - 1, w - 1]

    x = tf.concat([x, tf.zeros(padding_shape_h, x.dtype)], 2)
    x = tf.concat([x, tf.zeros(padding_shape_w, x.dtype)], 3)

    y = tf.concat([y, tf.zeros(padding_shape_h, y.dtype)], 2)
    y = tf.concat([y, tf.zeros(padding_shape_w, y.dtype)], 3)

    x = tf.cast(x, tf.complex64)
    y = tf.cast(y, tf.complex64)

    x = tf.signal.fft2d(x)
    y = tf.signal.fft2d(y)

    out = tf.math.conj(y) * x
    out = tf.signal.ifft2d(out)

    out = out[:, :, :h, :w]
    out = tf.math.real(out)

    if normalize:
        nx = w - tf.range(0, w)
        ny = h - tf.range(0, h)
        out /= tf.cast(ny[None, :] * nx[:, None], out.dtype)

    return out


def autocorrelation(x, normalize=False):
    """Calculates the autocorrelation of images with shape (batch_size, channels, height, width)."""
    shape = tf.shape(x)
    b, c, h, w = shape[0], shape[1], shape[2], shape[3],

    padding_shape_h = [b, c, h - 1, w]
    padding_shape_w = [b, c, 2 * h - 1, w - 1]

    x = tf.concat([x, tf.zeros(padding_shape_h, x.dtype)], 2)
    x = tf.concat([x, tf.zeros(padding_shape_w, x.dtype)], 3)

    x = tf.cast(x, tf.complex64)
    x = tf.signal.fft2d(x)
    x = tf.math.conj(x) * x
    x = tf.signal.ifft2d(x)

    x = x[:, :, :h, :w]
    x = tf.math.real(x)

    if normalize:
        nx = w - tf.range(0, w)
        ny = h - tf.range(0, h)
        x /= tf.cast(ny[None, :] * nx[:, None], x.dtype)

    return x


def indices(shape, dtype=tf.int32):
    result = tf.meshgrid(*[tf.range(s) for s in shape])
    result = tf.concat([tf.reshape(x, [-1, 1]) for x in result], axis=1)
    return tf.cast(result, dtype)


def triu_indices(n):
    ind = indices((n, n))
    mask = tf.greater(ind[:, 0], ind[:, 1])
    return tf.boolean_mask(ind, mask)


def xgcd(b, a):
    """Calculates the greatest common divisor of a and b and two additional numbers x and y such that d == x*b + y*a.

    Parameters
    ----------
    b, a: int
        Two numbers to calculate the gcd of.

    Returns
    -------
    tuple of int
        A tuple (d, x, y) such that d = gcd(b, a) and d == x*b + y*a.

    Source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm#Iterative_algorithm_3
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while a != 0:
        q, b, a = b // a, a, b % a
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.

    Source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = tf.abs(tf.sin(angle)), tf.abs(tf.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or tf.abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def random_vector(max_size, direction=None, n=2):
    if direction is None:
        direction = np.random.normal(0, 1, n)
    else:
        direction = np.array(direction)

    direction = direction / np.linalg.norm(direction)

    length = np.abs(np.random.normal(0.4, 0.1))
    length = np.maximum(np.minimum(length, 0.6), 0.2)

    return length * direction * max_size


def warp_polygon_coordinates(points, vertices, new_vertices):
    """Warps points within one polygon to points in another using a conformal mapping."""
    points = np.array(points)

    def subdivide(vert):
        v1 = vert
        v2 = np.roll(vert, -1, axis=0)
        lengths = np.sum((v2 -v1)**2, axis=1)
        i = np.argmax(lengths)
        return np.insert(vert, i+1, (v1[i] + v2[i])/2, axis=0)

    if len(vertices) == 3:
        vertices = subdivide(vertices)
    if len(new_vertices) == 3:
        new_vertices = subdivide(new_vertices)

    a1, b1, c1, d1 = vertices[:4]
    a2, b2, c2, d2 = new_vertices[:4]

    t1 = Triangle(Point(*a1), Point(*b1), Point(*d1))
    t2 = Triangle(Point(*b1), Point(*c1), Point(*d1))

    new_t1 = Triangle(Point(*a2), Point(*b2), Point(*d2))
    new_t2 = Triangle(Point(*b2), Point(*c2), Point(*d2))

    c1 = Circle(t1.circumcenter, np.linalg.norm(t1.circumcenter.normalized_array[:2] - a1))
    c2 = Circle(t2.circumcenter, np.linalg.norm(t2.circumcenter.normalized_array[:2] - b1))

    new_c1 = Circle(new_t1.circumcenter, np.linalg.norm(new_t1.circumcenter.normalized_array[:2] - a2))
    new_c2 = Circle(new_t2.circumcenter, np.linalg.norm(new_t2.circumcenter.normalized_array[:2] - b2))

    m1 = Transformation.from_points_and_conics(t1.vertices, new_t1.vertices, c1, new_c1)
    m2 = Transformation.from_points_and_conics(t2.vertices, new_t2.vertices, c2, new_c2)

    p = PointCollection(points, homogenize=True)
    ind1 = t1.contains(p)
    ind2 = t2.contains(p)

    p1 = p[ind1]
    p2 = p[ind2]

    new_p1 = m1 * p1
    new_p2 = m2 * p2

    points[ind1] = new_p1.normalized_array[:, :-1]
    points[ind2] = new_p2.normalized_array[:, :-1]

    return points
