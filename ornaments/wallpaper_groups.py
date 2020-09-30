import numpy as np
import tensorflow as tf
from geometer import Point, Rectangle, Line
from scipy.interpolate import griddata
from scipy.spatial import Voronoi

from .point_groups import point_groups
from .utils import xgcd, bilinear_sampler, polygon_mask, warp_polygon_coordinates, random_vector, rotation

group_labels = {
    "P1": 0,
    "P2": 1,
    "Pm": 2,
    "Pg": 3,
    "Cm": 4,
    "Pmm": 5,
    "Pmg": 6,
    "Pgg": 7,
    "Cmm": 8,
    "P4": 9,
    "P4m": 10,
    "P4g": 11,
    "P3": 12,
    "P3m1": 13,
    "P31m": 14,
    "P6": 15,
    "P6m": 16
}

lattice_labels = {
    'parallelogram': 0,
    'rhombic': 1,
    'rectangular': 2,
    'square': 3,
    'hexagonal': 4,
}

lattice_types = {
    "P1": 'parallelogram',
    "P2": 'parallelogram',
    "Pm": 'rectangular',
    "Pg": 'rectangular',
    "Cm": 'rhombic',
    "Pmm": 'rectangular',
    "Pmg": 'rectangular',
    "Pgg": 'rectangular',
    "Cmm": 'rhombic',
    "P4": 'square',
    "P4m": 'square',
    "P4g": 'square',
    "P3": 'hexagonal',
    "P3m1": 'hexagonal',
    "P31m": 'hexagonal',
    "P6": 'hexagonal',
    "P6m": 'hexagonal'
}

group_lattice_mapping = {group_labels[g]: lattice_labels[l] for g, l in lattice_types.items()}
group_lattice_mapping = tf.convert_to_tensor(list(group_lattice_mapping.values()))

control_points = {
    "P1": (0.5, 0.5),
    "P2": (0.5, 0.25),
    "Pm": (0.5, 0.25),
    "Pg": (0.5, 0.25),
    "Cm": (1, 0.5),
    "Pmm": (0.25, 0.25),
    "Pmg": (0.25, 0.25),
    "Pgg": (0.25, 0.25),
    "Cmm": (0.5, 1/3),
    "P4": (0.25, 0.25),
    "P4m": (1/3, 1/6),
    "P4g": (1/3, 1/6),
    "P3": (0.25, 0.5),
    "P3m1": (0.25, 0.25),
    "P31m": (0.5, 0.25),
    "P6": (1/6, 1/6),
    "P6m": (0.5, 1/3)
}

# subgroup hierarchy of wallpaper groups
group_hierarchy = {
    "P1": [],
    "P2": ["P1"],
    "Pm": ["P1"],
    "Pg": ["P1"],
    "Cm": ["P1"],
    "Pmm": ["P2", "Pm"],
    "Pmg": ["P2", "Pg"],
    "Pgg": ["P2"],
    "Cmm": ["P2", "Cm"],
    "P4": ["P2"],
    "P4m": ["P4", "Pmm", "Cmm"],
    "P4g": ["P4", "Pgg"],
    "P3": ["P1"],
    "P3m1": ["P3", "Cm"],
    "P31m": ["P3", "Cm"],
    "P6": ["P3", "P2"],
    "P6m": ["P6", "P3m1", "P31m", "Cmm"]
}


def get_subgroups(group):
    result = group_hierarchy[group]
    for g in result:
        for h in get_subgroups(g):
            if h in result:
                continue
            result.append(h)

    return result


def group_embedding(group):
    result = np.zeros(len(group_labels), dtype=int)
    for g in get_subgroups(group):
        result[group_labels[g]] = 1
    result[group_labels[group]] = 1
    return result


def group_name(label):
    for k, v in group_labels.items():
        if v == int(label):
            return k


def lattice_name(label):
    for k, v in lattice_labels.items():
        if v == int(label):
            return k


def lattice_matrix(t0, t1, t2):
    result = np.eye(3)
    result[:2, 0] = t1
    result[:2, 1] = t2
    result[:2, 2] = np.asarray(t0) - 0.5
    return result


def calculate_orbit(points, group, t0, t1, t2, a_min, a_max, b_min, b_max):
    points = np.asarray(points)

    transforms = point_groups[group.capitalize()]

    lattice_transform = lattice_matrix(t0, t1, t2)

    p_xy = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    p_ab = p_xy @ np.linalg.inv(lattice_transform).T
    p_ab = (p_ab.T / p_ab[:, 2]).T
    p_ab[:, :2] -= np.min(np.floor(p_ab[:, :2]), axis=0)

    result = []

    for transform in transforms:
        p_ab_t = p_ab @ transform.T
        for i in range(a_min, a_max+1):
            for j in range(b_min, b_max+1):
                p = p_ab_t + (i, j, 0)
                p = p @ lattice_transform.T
                p = (p.T[:2] / p[:, 2]).T

                result.append(p)

    return result


def calculate_visible_lattice_lines(t0, t1, t2, bounds):
    t0 = Point(*t0)
    t1 = Point(*t1)
    t2 = Point(*t2)

    x0, x1, y0, y1 = bounds
    viewport = Rectangle(Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1))

    result = []

    def add_parallels(start, end, direction):
        while True:
            line = Line(start, end)
            intersections = viewport.intersect(line)
            if len(intersections) == 0:
                break
            if len(intersections) > 1:
                 result.append(line)
            start += direction
            end += direction

    add_parallels(t0, t0+t1, t2)
    add_parallels(t0-t2, t0+t1-t2, -t2)
    add_parallels(t0, t0+t2, t1)
    add_parallels(t0-t1, t0+t2-t1, -t1)

    return result


def grid_points(t0, t1, t2, height, width, offset_height=0, offset_width=0):
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)
    a_min, a_max, b_min, b_max = calculate_ab_bounds(t0, t1, t2, height, width, offset_height, offset_width)

    result = []
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            p = t0 + a*t1 + b*t2

            if offset_width <= p[0] < offset_width+width and offset_height <= p[1] < offset_height+height:
                result.append(p)

    return np.array(result)


def calculate_ab_bounds(t0, t1, t2, height, width, offset_height=0, offset_width=0):
    vertices = np.array([(0, 0), (width, 0), (width, height), (0, height)])
    vertices = vertices + (offset_width, offset_height) - t0
    vertices = vertices @ np.linalg.inv(np.column_stack([t1, t2])).T
    a_min, b_min = np.min(np.floor(vertices).astype(int), axis=0)
    a_max, b_max = np.max(np.floor(vertices).astype(int), axis=0)
    return a_min, a_max, b_min, b_max


def calculate_visible_fundamental_cells(t0, t1, t2, group, height, width):
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)
    t0 = normalize_t0(t0, t1, t2, height, width)

    vertices = calculate_fundamental_region(group, t0, t1, t2)
    bounds = calculate_ab_bounds(t0, t1, t2, height, width)
    return calculate_orbit(vertices, group, t0, t1, t2, *bounds)


def normalize_t0(t0, t1, t2, height, width, offset_height=0, offset_width=0):
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)
    grid = grid_points(t0, t1, t2, height, width, offset_height, offset_width)

    if len(grid) == 0:
        return t0

    standard_cell = np.stack([grid, grid + t1, grid + t1 + t2, grid + t2], 1)
    xmin, ymin = np.min(standard_cell, axis=1).T
    xmax, ymax = np.max(standard_cell, axis=1).T

    rating = np.asarray(offset_width <= xmin).astype(int)
    rating += np.asarray(xmax < width).astype(int)
    rating += np.asarray(offset_height <= ymin).astype(int)
    rating += np.asarray(ymax < height).astype(int)
    t0_idx = np.argmax(rating)
    return grid[t0_idx]


def calculate_fundamental_region(group, t0, t1, t2):
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)

    control_point = control_points[group.capitalize()]
    control_point = t0 + control_point[0] * t1 + control_point[1] * t2 - 0.5

    w, h = np.ceil(np.max(np.abs([t1, t2]), axis=0))
    grid = grid_points(t0, t1, t2, 2*h+1, 2*w+1, t0[1]-h, t0[0]-w)
    t0_idx = np.where(np.all(grid == t0, axis=1))[0]

    # determine grid point closest to t0
    grid = np.delete(grid, t0_idx, axis=0)
    dist = np.sum((grid - control_point)**2, axis=1)
    c = grid[np.argmin(dist)]

    lattice_transform = lattice_matrix(t0, t1, t2)
    c_a, c_b = np.linalg.inv(lattice_transform).dot(np.append(c, [1])).astype(int)[:2]

    _, d_a, d_b = xgcd(c_b, c_a)
    d_a = -d_a

    d_prime = t0 + d_a*t1 + d_b*t2
    d = d_prime + np.ceil(-d_prime.dot(c)/c.dot(c))*c

    d_a, d_b = np.linalg.inv(lattice_transform).dot(np.append(d, [1])).astype(int)[:2]

    m_a = max(abs(c_a), abs(d_a), abs(c_a-d_a))
    m_b = max(abs(c_b), abs(d_b), abs(c_a-d_b))

    orbit = calculate_orbit([control_point], group, t0, t1, t2, -m_a-1, m_a+1, -m_b-1, m_b+1)
    orbit = np.concatenate(orbit, 0)
    control_index = np.where(np.all(np.isclose(orbit, control_point), axis=1))[0]

    if len(control_index) > 1:
        raise ValueError("The given control point lies on a symmetry axis/point of the pattern.")

    vor = Voronoi(orbit)
    region = vor.point_region[control_index[0]]
    vertices = vor.regions[region]
    return vor.vertices[vertices]


def extract_fundamental_region(image, group, t0, t1, t2):
    image = np.asarray(image)
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)

    h, w, c = image.shape
    vertices = calculate_fundamental_region(group, t0, t1, t2)
    mask = polygon_mask(vertices, (h, w)).T

    points = np.nonzero(mask)
    points = np.stack(points).T - t0
    values = image.transpose(1, 0, 2).reshape(-1, c)[mask.ravel()]

    return points, values


def random_lattice(group, max_size):
    lattice_type = lattice_types[group.capitalize()]

    t1 = random_vector(max_size)

    if lattice_type == 'parallelogram':
        r = rotation(np.random.rand()*4*np.pi/6 + np.pi/6)
        t2 = random_vector(max_size, r@t1)
    elif lattice_type == 'square':
        t2 = np.array([-t1[1], t1[0]])
    elif lattice_type == 'hexagonal':
        r = rotation(2*np.pi/3)
        t2 = r @ t1
    elif lattice_type == 'rectangular':
        g = point_groups[group.capitalize()]
        f = g[len(g)//2][:2, :2]
        _, v = np.linalg.eig(f)
        t1 = random_vector(max_size, v[:, 0])
        t2 = random_vector(max_size, [-t1[1], t1[0]])
    elif lattice_type == 'rhombic':
        g = point_groups[group.capitalize()]
        f = g[len(g) // 2][:2, :2]
        _, v = np.linalg.eig(f)
        r = rotation(np.random.rand() * np.pi / 6 + np.pi / 6)
        t1 = r @ random_vector(max_size, v[:, 0])
        t2 = f @ t1
    else:
        raise ValueError('unknown lattice type ' + str(lattice_type))

    t0 = np.random.rand(2)
    t0 = t0[0]*t1 + t0[0]*t2

    return t0, t1, t2


def create_wallpaper_pattern(fundamental_indices, fundamental_values, group, t0, t1, t2, height, width):
    fundamental_indices, fundamental_values = np.asarray(fundamental_indices), np.asarray(fundamental_values)
    fundamental_indices = fundamental_indices + t0
    c = fundamental_values.shape[-1]

    a_min, a_max, b_min, b_max = calculate_ab_bounds(t0, t1, t2, height, width)
    result_points = calculate_orbit(fundamental_indices, group, t0, t1, t2, a_min - 1, a_max + 1, b_min - 1, b_max + 1)

    orbit_length = len(result_points)
    result_points = np.concatenate(result_points, axis=0)
    indices = np.indices((width, height))
    indices = np.transpose(indices, (1, 2, 0))

    ind = np.all((result_points >= 0) & (result_points < (width, height)), axis=1)
    result_points = result_points[ind]

    result = []
    for idx in range(c):
        result_values = np.tile(fundamental_values[:, idx], orbit_length)
        result_values = result_values[ind]
        result.append(griddata(result_points, result_values, indices, method='nearest').T)

    result = np.stack(result, axis=2)
    return np.minimum(np.maximum(result, 0.0), 1.0)


def transform_wallpaper_pattern(image, group, new_group, t0, t1, t2):

    def filter_vertices(vert):
        result = [vert[0]]
        for v in vert[1:]:
            d = np.linalg.norm(v - result, axis=1)
            if np.any(d < 10):
                continue
            result.append(v)
        return result

    height, width, _ = image.shape
    new_t0, new_t1, new_t2 = random_lattice(new_group, (width, height))
    t0 = normalize_t0(t0, t1, t2, height, width)
    new_t0 = normalize_t0(new_t0, new_t1, new_t2, height, width)

    vertices = calculate_fundamental_region(group, t0, t1, t2)
    vertices = filter_vertices(vertices)
    new_vertices = calculate_fundamental_region(new_group, new_t0, new_t1, new_t2)
    new_vertices = filter_vertices(new_vertices)
    new_vertices = np.asarray(new_vertices)

    xmin, ymin = np.floor(np.min(new_vertices, axis=0)).astype(int)
    xmax, ymax = np.ceil(np.max(new_vertices, axis=0)).astype(int)
    h, w, c = ymax - ymin, xmax - xmin, image.shape[2]

    mask = polygon_mask(new_vertices - (xmin, ymin), (h, w))
    y, x = np.nonzero(mask)
    new_indices = np.stack([x, y], axis=1)
    result_indices = warp_polygon_coordinates(new_indices, new_vertices - (xmin, ymin), vertices)

    result_values = []
    indices = np.indices((image.shape[1], image.shape[0])).reshape(2, -1).T
    for idx in range(c):
        values = image.reshape(-1, c)[:, idx]
        result_values.append(griddata(indices, values, result_indices, method='nearest').T)

    result_values = np.stack(result_values, axis=1)

    new_image = create_wallpaper_pattern(new_indices, result_values, new_group, new_t0, new_t1, new_t2, height, width)
    new_lattice = np.concatenate([new_t0, new_t1, new_t2]).astype(np.float32)
    return new_image, new_lattice


@tf.function
def replicate_tiles(images, t0, t1, t2, new_shape=None, use_ab_coordinates=False):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)

    shape = tf.shape(images)
    batch_size = shape[0]

    if new_shape is None:
        new_shape = shape[1:3]

    height = new_shape[0]
    width = new_shape[1]

    transform = tf.stack([t1, t2], axis=2)
    transform = tf.expand_dims(transform, 1)

    inverse = [tf.stack([t2[:, 1], -t1[:, 1]], axis=1), tf.stack([-t2[:, 0], t1[:, 0]], axis=1)]
    inverse = tf.stack(inverse, axis=2)
    inverse = tf.expand_dims(inverse, 1)

    det = t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0]
    det = tf.reshape(det, [-1, 1, 1, 1])
    inverse = tf.math.divide_no_nan(inverse, det)

    X, Y = tf.meshgrid(tf.range(width), tf.range(height))
    indices = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y, [-1, 1])], axis=1)
    indices = tf.cast(indices, tf.float32)

    indices = tf.expand_dims(indices, 0) - tf.expand_dims(t0, 1)
    indices = tf.linalg.matvec(inverse, indices)

    indices = indices - tf.floor(indices)

    if use_ab_coordinates:
        indices = indices * tf.cast(shape[1:3], tf.float32)
    else:
        indices = tf.linalg.matvec(transform, indices)
        indices = indices + tf.expand_dims(t0, 1)

    indices = tf.reshape(indices, [batch_size, width, height, 2])

    return bilinear_sampler(images, indices[:, :, :, 0], indices[:, :, :, 1])


@tf.function
def extract_tiles(images, t0, t1, t2, height, width):
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = tf.convert_to_tensor(t2, dtype=tf.float32)
    batch_size = tf.shape(images)[0]

    transform = tf.stack([t1, t2], axis=2)
    transform = tf.expand_dims(transform, 1)

    X, Y = tf.meshgrid(tf.linspace(0.0, 1.0, width+1)[:-1], tf.linspace(0.0, 1.0, height+1)[:-1])
    indices = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y, [-1, 1])], axis=1)

    indices = tf.expand_dims(indices, 0)
    indices = tf.linalg.matvec(transform, indices)
    indices = indices + tf.expand_dims(t0, 1)

    indices = tf.reshape(indices, [batch_size, width, height, 2])

    return bilinear_sampler(images, indices[:, :, :, 0], indices[:, :, :, 1])
