import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imsave
from skimage.util import img_as_ubyte
from geometer import Polygon

from ornaments.point_groups import point_groups
from ornaments.wallpaper_groups import calculate_fundamental_region, create_wallpaper_pattern, normalize_t0, random_lattice
from ornaments.utils import resize_with_pad, polygon_mask


def mnist_tiles(digit):
    (images, labels), _ = tf.keras.datasets.mnist.load_data()
    images = np.expand_dims(images, 3)
    return images[labels == digit]


def random_tile(height, width):
    x = np.linspace(0.0, 1.0, width).reshape((1, width, 1))
    y = np.linspace(0.0, 1.0, height).reshape((height, 1, 1))

    functions = [(0, lambda: np.random.rand(1, 1, 3)),
                 (0, lambda: x),
                 (0, lambda: y),
                 (1, np.sin),
                 (1, np.cos),
                 (1, lambda a: np.exp(np.minimum(a, 2))),
                 (1, lambda a: np.log(np.maximum(a, 0.1))),
                 (2, np.add),
                 (2, np.subtract),
                 (2, np.multiply),
                 (2, lambda a, b: a / np.maximum(b, 1e-3))]

    mindepth = 2
    maxdepth = 20

    def build_img(depth=0):
        funcs = functions
        if depth == maxdepth:
            funcs = [f for f in funcs if f[0] == 0]
        if depth < mindepth:
            funcs = [f for f in funcs if f[0] > 0]

        idx = np.random.randint(len(funcs))
        n, f = funcs[idx]
        args = [build_img(depth + 1) for _ in range(n)]

        return f(*args)

    result = build_img()

    if result.shape[0] != height:
        result = result + y

    if result.shape[1] != width:
        result = result + x

    result = np.abs(result)

    return result / np.maximum(np.max(result), 1)


def create_ornament(tile, group, t0, t1, t2, height, width):
    t0, t1, t2 = np.array(t0), np.array(t1), np.array(t2)
    t0 = normalize_t0(t0, t1, t2, height, width)

    vertices = calculate_fundamental_region(group, t0, t1, t2)

    center = Polygon(np.append(vertices, np.ones((vertices.shape[0], 1)), axis=1)).centroid.normalized_array[:2]

    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)

    h = 2*max(ymax-center[1], center[1]-ymin)+1
    w = 2*max(xmax-center[0], center[0]-xmin)+1
    c = tile.shape[2]

    offset = center - (w/2, h/2)
    shape = np.ceil((h, w)).astype(int)

    mask = polygon_mask(vertices - offset, shape).flatten()

    points = np.indices(shape)
    points = np.reshape(points, (2, -1)).T
    points = points[mask, ::-1] + offset

    tile = resize_with_pad(tile, *shape)
    values = tile.reshape(-1, c)[mask]

    return create_wallpaper_pattern(points, values, group, t0, t1, t2, height, width)


def generate_mnist_ornaments(height, width, output_dir='.', digit=4):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tiles = mnist_tiles(digit)
    data = []

    for tile in tiles:
        for group in point_groups.keys():

            while True:
                t0, t1, t2 = random_lattice(group, (width, height))

                try:
                    ornament = create_ornament(tile, group, t0, t1, t2, height, width)
                except ValueError:
                    pass
                else:
                    break

            fn = 'ornament_{}.png'.format(str(len(data)))
            imsave(os.path.join(output_dir, fn), img_as_ubyte(ornament))
            data.append((fn, group, *np.concatenate([t0, t1, t2])))

    df = pd.DataFrame(data, columns=['filename', 'group', 't0_0', 't0_1', 't1_0', 't1_1', 't2_0', 't2_1'])
    df.to_csv(os.path.join(output_dir, 'generated_groups.csv'), index=False)


def merge_directories(input_dirs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dfs = []
    for i, img_path in enumerate(input_dirs):
        df = pd.read_csv(os.path.join(img_path, 'generated_groups.csv'))
        new_filenames = str(i) + '_' + df['filename']
        for fn, new_fn in zip(df['filename'], new_filenames):
            subprocess.call(['cp', os.path.join(img_path, fn), os.path.join(output_dir, new_fn)])

        df['filename'] = new_filenames
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(output_dir, 'generated_groups.csv'), index=False)


if __name__ == "__main__":
    digits = [1, 2, 4, 5, 6, 7]

    with ProcessPoolExecutor(max_workers=6) as executor:
        for digit in digits:
            executor.submit(generate_mnist_ornaments, 256, 256, 'data/images_generated_' + str(digit), digit)

    merge_directories(['data/images_generated_' + str(d) for d in digits], 'data/images_generated')
