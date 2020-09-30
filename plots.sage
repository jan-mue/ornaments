import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import griddata
from geometer import Point, Rectangle, Polygon
import matplotlib.pyplot as plt

from run_lattice_extraction import LatticeExtraction
from ornaments.modeling.layers import autocorrelation
from ornaments.wallpaper_groups import point_groups, control_points, create_wallpaper_pattern, grid_points, calculate_ab_bounds, \
    calculate_orbit, calculate_fundamental_region, calculate_visible_lattice_lines, polygon_mask, warp_polygon_coordinates, transform_wallpaper_pattern


def round_matrix(m, digits):
    return Matrix([[int(x) if round(x, digits) == int(x) else round(x, digits) for x in row] for row in m])


def wallpaper_group_latex():
    s = [r"\begin{align*}"]
    for group, matrices in point_groups.items():
        s.append(r"\mathrm{" + group.lower() + r"}\qquad")
        s.append("&")
        if group == "P6m":
            s.extend(latex(round_matrix(m, 1)) + r"\quad" for m in matrices[:6])
            s.append(r"\\&")
            s.extend(latex(round_matrix(m, 1)) + r"\quad" for m in matrices[6:])
        else:
            s.extend(latex(round_matrix(m, 1)) + r"\quad" for m in matrices)
        s.append(r"\\")
    s.append(r"\end{align*}")
    return "".join(s).replace("0.0", "0").replace("1.0", "1")


def control_points_latex():
    s = [r"\begin{align*}"]
    for i in range(6):
        for j in range(3*i, min(3*(i+1), 17)):
            group, point = list(control_points.items())[j]
            s.append(r"\mathrm{" + group.lower() + r"}\qquad")
            point = vector(QQ, point)
            s.append("&" + latex(point) + "&")
        s.append(r"\\")
    s.append(r"\end{align*}")
    return "".join(s).replace('frac', 'tfrac')


def axes(x_min=-1.5, x_max=1.5, y_min=-1, y_max=1):
    return arrow((0, y_min), (0, y_max), color='black') + arrow((x_min, 0), (x_max, 0), color='black')


def grid(t1, t2, x_min=-1.5, x_max=1.5, y_min=-1, y_max=1):
    points = grid_points((0, 0), t1, t2, y_max-y_min, x_max-x_min, offset_height=y_min, offset_width=x_min)
    return scatter_plot(points, markersize=30, facecolor='black')


def create_polygon_pattern(vertices, group, t0, t1, t2, height, width, show_lattice=False, show_fundamental_regions=False):
    vertices = np.asarray(vertices) + t0

    a_min, a_max, b_min, b_max = calculate_ab_bounds(t0, t1, t2, height, width)
    result_points = calculate_orbit(vertices, group, t0, t1, t2, a_min, a_max, b_min-1, b_max+1)

    result = Graphics()
    for v in result_points:
        result += polygon(v, rgbcolor=(0,0,0))

    if show_lattice is True:
        result += lattice(t0, t1, t2, x_min=0, x_max=width, y_min=0, y_max=height, color='black')

    if show_fundamental_regions is True:
        result += fundamental_regions(group, t0, t1, t2, height, width, color='grey')

    return result


def lattice(t0, t1, t2, x_min=-1.5, x_max=1.5, y_min=-1, y_max=1, color='grey'):
    t0, t1, t2 = np.array(t0), np.array(t1), np.array(t2)
    lines = calculate_visible_lattice_lines(t0, t1, t2, (x_min, x_max, y_min, y_max))
    viewport = Rectangle(Point(x_min, y_min), Point(x_max, y_min), Point(x_max, y_max), Point(x_min, y_max))

    points = []
    for l in lines:
        a, b = viewport.intersect(l)
        a, b = a.normalized_array[:2], b.normalized_array[:2]
        points.append([a, b]) 

    result = Graphics()
    for pts in points:
        result += line(pts, color=color)
    return result


def fundamental_regions(group, t0, t1, t2, height, width, color='red'):
    vertices = calculate_fundamental_region(group, t0, t1, t2)

    a_min, a_max, b_min, b_max = calculate_ab_bounds(t0, t1, t2, height, width)
    result_points = calculate_orbit(vertices, group, t0, t1, t2, a_min-1, a_max+2, b_min-1, b_max+2)

    result = Graphics()
    for v in result_points:

        if np.min(v[:, 0]) > width or np.min(v[:, 1]) > height or np.max(v[:, 0]) < 0 or np.max(v[:, 1]) < 0:
            continue

        result += polygon(v, fill=False, color=color)

    return result


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0.8, 1.0, N+4)
    return mycmap


def image_autocorrelation():
    img = plt.imread("gfx/example.png").astype(float)
    img = np.expand_dims(img, 0)

    img = np.transpose(img, [0, 3, 1, 2])
    img = autocorrelation(img)
    img = np.sum(img, axis=1)

    return matrix_plot(img[0], frame=False, cmap='gray')


def image_with_voting_space():
    img = plt.imread("gfx/example.png").astype(float)
    votes = plt.imread("gfx/example_votes.png").astype(float)
    mycmap = transparent_cmap(plt.get_cmap('plasma'))
    return matrix_plot(img, frame=False) + matrix_plot(votes, cmap=mycmap, frame=False)


def image_with_extracted_lattice():
    img = plt.imread("gfx/example.png").astype(float)
    model = LatticeExtraction(img.shape)
    # t0, t1, t2 = model(np.expand_dims(img, 0))[0]
    t0, t1, t2 = (0, 0), (128, 0), (0, 128)
    return matrix_plot(img, frame=False) + lattice(t0, t1, t2, x_min=0, x_max=img.shape[1], y_min=0, y_max=img.shape[0], color='red')


def image_with_lattice(filename, t0, t1, t2, color='red'):
    img = plt.imread(filename).astype(float)
    return matrix_plot(img, frame=False) + lattice(t0, t1, t2, x_min=0, x_max=img.shape[1], y_min=0, y_max=img.shape[0], color=color)


def image_with_fundamental_regions(filename, group, t0, t1, t2, color='red'):
    img = plt.imread(filename).astype(float)
    return matrix_plot(img, frame=False) + fundamental_regions(group, t0, t1, t2, img.shape[1], img.shape[0], color=color)


def extracted_fundamental_region(filename, group, t0, t1, t2):
    img = plt.imread(filename).astype(float)
    t0, t1, t2 = np.asarray(t0), np.asarray(t1), np.asarray(t2)

    vertices = calculate_fundamental_region(group, t0, t1, t2)
    xmin, ymin = np.floor(np.min(vertices, axis=0)).astype(int)
    xmax, ymax = np.ceil(np.max(vertices, axis=0)).astype(int)
    h, w = ymax-ymin, xmax-xmin
    mask = polygon_mask(vertices-(xmin, ymin), (h, w))
    img = np.concatenate([img[ymin:ymax, xmin:xmax], np.ones((h, w, 1), img.dtype)], axis=2)
    img = img*np.expand_dims(mask, 2)

    return matrix_plot(img, frame=False)


def transformed_fundamental_region(filename, group, new_group, t0, t1, t2, new_t0, new_t1, new_t2):
    img = plt.imread(filename).astype(float)

    vertices = calculate_fundamental_region(group, t0, t1, t2)
    new_vertices = calculate_fundamental_region(new_group, new_t0, new_t1, new_t2)

    xmin, ymin = np.floor(np.min(new_vertices, axis=0)).astype(int)
    xmax, ymax = np.ceil(np.max(new_vertices, axis=0)).astype(int)
    h, w, c = ymax-ymin, xmax-xmin, img.shape[2]

    result = np.zeros((h, w, c+1))

    mask = polygon_mask(new_vertices-(xmin, ymin), (h, w))
    y, x = np.nonzero(mask)
    new_indices = np.stack([x, y], axis=1)
    result_indices = warp_polygon_coordinates(new_indices, new_vertices-(xmin, ymin), vertices)
    
    indices = np.indices((img.shape[1], img.shape[0])).reshape(2, -1).T
    for idx in range(c):
        values = img.reshape(-1, c)[:, idx]
        result_values = griddata(indices, values, result_indices, method='nearest').T
        result[mask, idx] = result_values

    result[mask, 3] = 1

    return matrix_plot(result, frame=False)


def transformed_wallpaper_pattern(filename, group, new_group, t0, t1, t2, new_t0, new_t1, new_t2):
    img = plt.imread(filename).astype(float)
    img = transform_wallpaper_pattern(img, group, new_group, t0, t1, t2, new_t0, new_t1, new_t2)
    return matrix_plot(img, frame=False)


def tile_with_fundamental_region(filename, group, t0, t1, t2, color='red'):
    img = plt.imread(filename).astype(float)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    vertices = calculate_fundamental_region(group, t0, t1, t2)

    center = Polygon(np.append(vertices, np.ones((vertices.shape[0], 1)), axis=1)).centroid.normalized_array[:2]

    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)

    h = 2*max(ymax-center[1], center[1]-ymin)+1
    w = 2*max(xmax-center[0], center[0]-xmin)+1
    c = img.shape[2]

    offset = center - (w/2, h/2)
    shape = np.ceil((h, w)).astype(int)

    img = tf.image.resize_with_pad(img, *shape)

    if img.shape[-1] == 1:
        img = np.tile(img, (1, 1, 3))

    return matrix_plot(img, frame=False) + polygon(vertices - offset, fill=False, color=color)


def mnist_wallpaper_pattern(filename, group, t0, t1, t2, height, width):
    img = plt.imread(filename).astype(float)

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    vertices = calculate_fundamental_region(group, t0, t1, t2)

    center = Polygon(np.append(vertices, np.ones((vertices.shape[0], 1)), axis=1)).centroid.normalized_array[:2]

    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)

    h = 2*max(ymax-center[1], center[1]-ymin)+1
    w = 2*max(xmax-center[0], center[0]-xmin)+1
    c = img.shape[2]

    offset = center - (w/2, h/2)
    shape = np.ceil((h, w)).astype(int)

    mask = polygon_mask(vertices - offset, shape).flatten()

    points = np.indices(shape)
    points = np.reshape(points, (2, -1)).T
    points = points[mask, ::-1] + offset

    img = tf.image.resize_with_pad(img, *shape)
    values = np.reshape(img, (-1, c))[mask]

    result = create_wallpaper_pattern(points, values, group, t0, t1, t2, height, width)

    if result.shape[-1] == 1:
        result = np.tile(result, (1, 1, 3))

    return matrix_plot(result, frame=False) + fundamental_regions(group, t0, t1, t2, height, width)


def plot_csv(*filenames, kind='bar', legend=False, scale=0.4, **kwargs):
    df = pd.read_csv(filenames[0], **kwargs)
    for filename in filenames[1:]:
        df = pd.merge(df, pd.read_csv(filename, **kwargs), on=df.columns[0])

    variables = df.columns[1:1+len(filenames)]
    df.plot(df.columns[0], variables, kind=kind, legend=legend)
    fn, ext = os.path.splitext(os.path.basename(filenames[0]))
    fn = "sage-plots-for-my-thesis.tex/" + fn + "_plot.pdf"
    plt.savefig(fn, format="pdf")
    return r"\includegraphics[scale=" + str(scale) + "]{" + fn + "}"
