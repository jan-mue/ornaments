import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import imsave
from skimage.util import img_as_ubyte

from ornaments.data_loading.preprocessing import load_from_file
from ornaments.data_loading.split_data import filter_data
from ornaments.data_loading.download_images import load_ornaments_df
from ornaments.wallpaper_groups import group_labels, group_name


def save_image(img, filename):
    imsave(filename, img_as_ubyte(img))


def export_statistics():
    df = pd.DataFrame(filter_data(load_ornaments_df(''), 'data/'), columns=['Filename', 'Group'])
    corrections = pd.read_csv('corrections.csv')
    corrections.columns = ['Filename', 'Group']
    df['Group'] = df['Group'].str.lower()
    corrections['Group'] = corrections['Group'].str.lower()

    df.groupby('Group').Filename.nunique().to_csv('statistics.csv')

    df = df.set_index('Filename')
    corrections = corrections.set_index('Filename')
    df.update(corrections)

    df.reset_index().groupby('Group').Filename.nunique().to_csv('statistics_corrected.csv')


if __name__ == '__main__':

    all_groups = set(group_labels.values())
    train_ds = load_from_file('data/train.csv', 'data/images_train/', height=512, width=512, labels='all')

    lattices = []
    for image, label, lattice in train_ds.skip(32):
        label = int(label)
        if label in all_groups:
            save_image(image, group_name(label) + '.png')
            lattices.append([group_name(label)] + np.asarray(lattice).tolist())
            all_groups.remove(label)
        if len(all_groups) == 0:
            break

    pd.DataFrame(lattices, columns=['group', 't0_0', 't0_1', 't1_0', 't1_1', 't2_0', 't2_1']).to_csv('lattices.csv')

    (images, labels), _ = tf.keras.datasets.mnist.load_data()

    save_image(images[42], 'mnist_example.png')

    export_statistics()
