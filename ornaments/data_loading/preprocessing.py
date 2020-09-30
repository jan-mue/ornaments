import os

import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils import random_choice, rotate_with_crop
from ..wallpaper_groups import group_labels, group_embedding, group_name, normalize_t0, get_subgroups, transform_wallpaper_pattern


def load_from_file(csv_file, image_dir, height, width, n_tiles=4, channels=3, labels='groups', mean=0, std=1,
                   random_rotation=False, random_translation=False, augment_groups=False):
    df = pd.read_csv(csv_file)
    lattices = df[['t0_0', 't0_1', 't1_0', 't1_1', 't2_0', 't2_1']].to_numpy().astype('float32')

    ratio = width / height

    def load_and_preprocess(filename, group, lattice):
        image = tf.io.read_file(image_dir + filename)
        image = tf.image.decode_png(image, channels=channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        t0, t1, t2 = tf.unstack(tf.reshape(lattice, (3, 2)))

        t = random_choice(n_tiles) if isinstance(n_tiles, (list, tuple)) else int(n_tiles)

        if t > 1:
            shape = tf.shape(image)
            image = tf.tile(image, (t+1, t+1, 1))
            tiled_height, tiled_width = tf.cast(shape[0]*t, tf.float32), tf.cast(shape[1]*t, tf.float32)
            tiled_ratio = tiled_width / tiled_height

            new_height = tiled_height if tiled_ratio > ratio else tiled_width / ratio
            new_width = tiled_width if tiled_ratio <= ratio else ratio * tiled_height

            if random_translation:
                offset_y = tf.random.uniform([], 0, shape[0], dtype=tf.int32)
                offset_x = tf.random.uniform([], 0, shape[1], dtype=tf.int32)
            else:
                offset_y = 0
                offset_x = 0

            # TODO: somehow scaling & offset do not work with random values
            scale_x, scale_y = width / new_width, height / new_height
            t0 = (t0 + (offset_x, offset_y)) * (scale_x, scale_y)
            t1 = t1 * (scale_x, scale_y)
            t2 = t2 * (scale_x, scale_y)

            image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32))

        if random_rotation:
            # TODO: rotate lattice vectors
            r = tf.random.uniform([], 0, 2*np.pi)
            image = rotate_with_crop(image, r)

        image = tf.image.resize(image, [height, width], method='nearest')

        if augment_groups:
            out_types = [tf.float32, tf.int32, tf.float32]
            patterns, groups, lattices = tf.numpy_function(extend_patterns, [image, group, t0, t1, t2], out_types)
            patterns = tf.data.Dataset.from_tensor_slices(patterns)
            groups = tf.data.Dataset.from_tensor_slices(groups)
            lattices = tf.data.Dataset.from_tensor_slices(lattices)
            return tf.data.Dataset.zip((patterns, groups, lattices))

        # normalize image
        image = (image - mean) / std

        t0 = tf.numpy_function(normalize_t0, [t0, t1, t2, height, width], t0.dtype)
        lattice = tf.concat([t0, t1, t2], 0)

        if labels == 'group_embedding':
            group = group_embedding(group)

        return image, group, lattice

    def extend_patterns(image, group, t0, t1, t2):
        group = group_name(group)

        patterns = []
        lattices = []
        groups = []
        for new_group in group_labels.keys():
            if new_group == group:
                patterns.append(image)
                lattices.append(np.concatenate([t0, t1, t2]))
                groups.append(group_labels[group])
                continue

            if group not in get_subgroups(new_group):
                continue

            new_image, new_lattice = transform_wallpaper_pattern(image, group, new_group, t0, t1, t2)
            patterns.append(new_image)
            lattices.append(new_lattice)
            groups.append(group_labels[new_group])

        return patterns, np.array(groups, dtype=np.int32), lattices

    path_ds = tf.data.Dataset.from_tensor_slices(df["filename"])
    lattice_ds = tf.data.Dataset.from_tensor_slices(lattices)
    group_ds = tf.data.Dataset.from_tensor_slices(df["group"].map(lambda x: group_labels[x]))

    if augment_groups:
        ds = tf.data.Dataset.zip((path_ds, group_ds, lattice_ds)).interleave(load_and_preprocess, cycle_length=1,
                                                                             block_length=17, num_parallel_calls=-1)
    else:
        ds = tf.data.Dataset.zip((path_ds, group_ds, lattice_ds)).map(load_and_preprocess, num_parallel_calls=-1)

    if labels == 'all':
        return ds

    image_ds = ds.map(lambda a, b, c: a)
    label_ds = ds.map(lambda a, b, c: b)
    lattice_ds = ds.map(lambda a, b, c: c)

    if labels == 'lattice':
        return tf.data.Dataset.zip((image_ds, lattice_ds))

    elif labels in ['groups', 'group_embedding']:
        return tf.data.Dataset.zip((image_ds, label_ds))

    else:
        raise ValueError('Only groups, group embeddings and lattice labels are supported.')


def load_from_tfrecord(records_dir, labels='groups', height=None, width=None, use_lattice=True, mean=0, std=1,
                       shuffle_files=False, random_rotation=False, random_hue=False):
    filenames = tf.data.Dataset.list_files(os.path.join(records_dir, '*'), shuffle=shuffle_files)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=8)

    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
        'image/class/group': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros((), dtype=tf.int64))
    }

    if use_lattice is True:
        feature_description['image/class/lattice'] = tf.io.FixedLenFeature([6], tf.float32, default_value=tf.zeros(6, dtype=tf.float32))

    def parse_function(record):
        features = tf.io.parse_single_example(record, feature_description)
        image = tf.image.decode_png(features['image/encoded'])
        image = tf.image.convert_image_dtype(image, tf.float32)

        if labels == 'groups':
            label = tf.cast(features['image/class/group'], tf.int32)
        elif labels == 'group_embedding':
            label = tf.cast(features['image/class/group'], tf.int32)
            label = tf.py_function(lambda x: group_embedding(group_name(x)), [label], tf.int32)
        elif labels == 'lattice':
            label = features['image/class/lattice']
        else:
            raise ValueError('Only groups, group embeddings and lattice labels are supported.')

        if height is not None and width is not None:
            image = tf.image.resize(image, [height, width], method='nearest')

        if random_rotation:
            r = random_choice([0, 1, 2, 3])
            image = tf.image.rot90(image, r)

        if random_hue:
            image = tf.image.random_hue(image, 0.5)

        # normalize image
        image = (image - mean) / std

        return image, label

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def load_generated_data(data_dir='data', height=256, width=256, labels='groups'):
    mean = tf.convert_to_tensor([0.17232579])
    std = tf.convert_to_tensor([0.3044424])

    train_ds = load_from_tfrecord(os.path.join(data_dir, 'images_generated_records'), height=height, width=width, labels=labels, mean=mean, std=std, random_rotation=True)
    test_ds = load_from_tfrecord(os.path.join(data_dir, 'images_generated_records_test'), height=height, width=width, labels=labels, mean=mean, std=std)

    return train_ds, test_ds


def calculate_mean_and_std(ds):
    count = 0
    mean = 0.0
    for image, label in ds:
        mean += tf.reduce_mean(image, [0, 1])
        count += 1

    mean /= tf.cast(count, tf.float32)

    variance = 0.0
    for image, label in ds:
        variance += tf.reduce_mean((image - mean)**2, [0, 1])

    variance /= tf.cast(count, tf.float32)

    return mean, tf.sqrt(variance)


def load_data(data_dir='data', height=256, width=256, labels='groups'):
    mean = tf.convert_to_tensor([0.35580543, 0.3052868, 0.3320374])
    std = tf.convert_to_tensor([0.311617, 0.2857735, 0.2987481])

    train_csv, test_csv = os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'test.csv')
    train_dir, test_dir = os.path.join(data_dir, 'images_train/'), os.path.join(data_dir, 'images_test/')
    train_ds = load_from_file(train_csv, train_dir, height, width, labels=labels, n_tiles=[3, 4, 5],
                              mean=mean, std=std, random_rotation=True, random_translation=True)
    test_ds = load_from_file(test_csv, test_dir, height, width, labels=labels, mean=mean, std=std)

    return train_ds, test_ds


def to_tfrecord(dataset, name, output_dir='data', num_shards=32, use_lattice=True):
    def int64_feature(values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def float_list_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def bytes_feature(values):
        if isinstance(values, type(tf.constant(0))):
            values = values.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def serialize_example(*args):

        image = tf.image.convert_image_dtype(args[0], tf.uint8)

        feature = {
            'image/encoded': bytes_feature(tf.image.encode_png(image)),
            'image/format': bytes_feature(b'png'),
            'image/class/group': int64_feature(args[1])
        }

        if use_lattice is True:
            feature['image/class/lattice'] = float_list_feature(args[2])

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def tf_serialize_example(*args):
        tf_string = tf.py_function(serialize_example, args, tf.string)
        return tf.reshape(tf_string, ())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = dataset.map(tf_serialize_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for s in range(num_shards):
        shard = dataset.shard(num_shards, s)
        filename = '%s-%.5d-of-%.5d' % (name, s, num_shards)
        writer = tf.data.experimental.TFRecordWriter(os.path.join(output_dir, filename))
        writer.write(shard)
