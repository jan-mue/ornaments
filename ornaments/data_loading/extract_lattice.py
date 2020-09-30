import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from skimage.io import imread

from ornaments.utils import rotation
from ornaments.wallpaper_groups import lattice_types


def load_and_extract_lattice(filename, group):
    image = imread(filename)
    h, w, c = image.shape
    lattice_type = lattice_types[group.capitalize()]
    t0 = (0, 0)

    if group.capitalize() == 'P31m' or '1384274104.png' in filename:
        t0 = (w / 2, (h+1) / 6)
    elif group.capitalize() == 'P4g' and '1393137911.png' not in filename:
        t0 = (w / 2, 0.0)
    elif group.capitalize() == 'P4m':
        t0 = (w / 2, (h + 1) / 2)

    if (group.capitalize() == 'P3' or '1395432421.png' in filename) and w != h:
        t1 = (w / 2, (h + 1) / 2)
        t2 = (0, h)
    elif lattice_type == 'hexagonal':
        # add one to height because last row of pixels is missing
        t2 = (w / 2, (h+1) / 2)
        r = rotation(2*np.pi/3)
        t1 = r @ t2
    else:
        t1, t2 = (w, 0), (0, h)

    if group.capitalize() == 'P6m' and (h, w) == (1280, 739):
        t1[1] = 0.0

    if group.capitalize() == 'Pm':
        t1, t2 = t2, t1

    def r(x):
        x = np.asarray(x)
        return np.where(x < 0, np.floor(x), np.ceil(x)).astype(int)

    t0, t1, t2 = r(t0), r(t1), r(t2)

    return os.path.basename(filename), t0[0], t0[1], t1[0], t1[1], t2[0], t2[1]


if __name__ == "__main__":
    corrections = pd.read_csv('corrections.csv').set_index('filename')
    train_df = pd.read_csv('data/train.csv', index_col=0)
    test_df = pd.read_csv('data/test.csv', index_col=0)

    train_df = train_df.set_index('filename')
    test_df = test_df.set_index('filename')

    train_df.update(corrections)
    test_df.update(corrections)

    with ProcessPoolExecutor(max_workers=8) as executor:
        fn_generator = ('data/images_train/' + fn for fn, row in train_df.iterrows())
        g_generator = (row['group'] for fn, row in train_df.iterrows())
        train_lattice = executor.map(load_and_extract_lattice, fn_generator, g_generator)

        fn_generator = ('data/images_test/' + fn for fn, row in test_df.iterrows())
        g_generator = (row['group'] for fn, row in test_df.iterrows())
        test_lattice = executor.map(load_and_extract_lattice, fn_generator, g_generator)

    train_lattice = pd.DataFrame(train_lattice, columns=['filename', 't0_0', 't0_1', 't1_0', 't1_1', 't2_0', 't2_1'])
    test_lattice = pd.DataFrame(test_lattice, columns=['filename', 't0_0', 't0_1', 't1_0', 't1_1', 't2_0', 't2_1'])

    train_df = train_df.merge(train_lattice, on='filename')
    test_df = test_df.merge(test_lattice, on='filename')

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

