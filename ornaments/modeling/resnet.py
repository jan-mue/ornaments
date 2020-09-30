import h5py
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, ReLU, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, AvgPool2D
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group_by_name
from tensorflow.python.keras.applications.resnet import WEIGHTS_HASHES, BASE_WEIGHTS_PATH


def load_imagenet_weights(resnet_model, model_name, include_top=True, skip_mismatch=False):
    def get_layers(layers):
        layers_flat = []
        for layer in layers:
            try:
                layers_flat.extend(get_layers(layer.layers))
            except AttributeError:
                layers_flat.append(layer)
        return layers_flat

    if include_top:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
        file_hash = WEIGHTS_HASHES[model_name][0]
    else:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name][1]
    weights_path = data_utils.get_file(file_name, BASE_WEIGHTS_PATH + file_name, cache_subdir='models',
                                       file_hash=file_hash)

    with h5py.File(weights_path, 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        load_weights_from_hdf5_group_by_name(f, get_layers(resnet_model.layers), skip_mismatch=skip_mismatch)


class BasicBlock(Model):

    def __init__(self, depth, expansion=1, kernel_size=3, strides=1, groups=1,
                 conv_shortcut=False, avg_down=False, avd=False, use_bias=True, name=None):
        super(BasicBlock, self).__init__(name=name)

        self.conv1 = Sequential([
            Conv2D(depth, kernel_size, strides=strides if not avd else 1, padding='same', use_bias=use_bias,
                   kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4), name=name + '_1_conv'),
            BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn'),
            ReLU(name=name + '_1_relu')
        ], name='1')

        if avd:
            self.conv2.add(AvgPool2D(kernel_size, strides=strides, padding='same', name=name + '_1_avg_pool'))

        self.conv2 = Sequential([
            Conv2D(depth, kernel_size, use_bias=use_bias, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
                   padding='same', name=name + '_2_conv'),
            BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')
        ], name='2')

        if strides > 1 and avg_down:
            self.shortcut = Sequential([
                AvgPool2D(strides, strides=strides, padding='same', name=name + '_0_avg_pool'),
                Conv2D(depth, 1, strides=1, use_bias=use_bias, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
                       name=name + '_0_conv'),
                BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')
            ], name='0')
        elif strides > 1 or conv_shortcut:
            self.shortcut = Sequential([
                Conv2D(depth, 1, strides=strides, use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4), name=name + '_0_conv'),
                BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')
            ], name='0')
        else:
            self.shortcut = tf.identity

        self.relu = ReLU(name=name + '_out')

    def call(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        outputs += self.shortcut(inputs)
        return self.relu(outputs)


class BottleneckBlock(Model):

    def __init__(self, depth, expansion=4, kernel_size=3, strides=1, groups=1,
                 conv_shortcut=False, avg_down=False, avd=False, use_bias=True, name=None):
        super(BottleneckBlock, self).__init__(name=name)

        self.conv1 = Sequential([
            Conv2D(depth, 1, use_bias=use_bias, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
                   name=name + '_1_conv'),
            BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn'),
            ReLU(name=name + '_1_relu')
        ], name='1')

        self.conv2 = Sequential([
            Conv2D(depth, kernel_size, strides=strides if not avd else 1, use_bias=use_bias,
                   kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4), groups=groups,
                   padding='same', name=name + '_2_conv'),
            BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn'),
            ReLU(name=name + '_2_relu')
        ], name='2')

        if avd:
            self.conv2.add(AvgPool2D(kernel_size, strides=strides, padding='same', name=name + '_2_avg_pool'))

        self.conv3 = Sequential([
            Conv2D(depth * expansion, 1, use_bias=use_bias, kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
                   name=name + '_3_conv'),
            BatchNormalization(epsilon=1.001e-5, name=name + '_3_bn'),
        ], name='3')

        if strides > 1 and avg_down:
            self.shortcut = Sequential([
                AvgPool2D(strides, strides=strides, padding='same', name=name + '_0_avg_pool'),
                Conv2D(depth * expansion, 1, strides=1, use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4), name=name + '_0_conv'),
                BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')
            ], name='0')
        elif strides > 1 or conv_shortcut:
            self.shortcut = Sequential([
                Conv2D(depth * expansion, 1, strides=strides, use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4), name=name + '_0_conv'),
                BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')
            ], name='0')
        else:
            self.shortcut = tf.identity

        self.relu = ReLU(name=name + '_out')

    def call(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        outputs += self.shortcut(inputs)
        return self.relu(outputs)


class ResNet(Model):

    def __init__(self, layers, num_classes=1000, block=BottleneckBlock, expansion=4, deep_stem=False, stem_depth=64, base_depth=64,
                 avg_down=False, avd=False, groups=1, use_bias=True, include_top=True, classifier_activation='softmax'):
        super(ResNet, self).__init__()

        if deep_stem:
            self.stem = Sequential([
                Conv2D(stem_depth, 3, strides=2, padding='same', use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4)),
                BatchNormalization(epsilon=1.001e-5),
                ReLU(),
                Conv2D(stem_depth, 3, padding='same', use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4)),
                BatchNormalization(epsilon=1.001e-5),
                ReLU(),
                Conv2D(stem_depth * 2, 3, padding='same', use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4)),
                BatchNormalization(epsilon=1.001e-5),
                ReLU()
            ])
        else:
            self.stem = Sequential([
                Conv2D(stem_depth, 7, strides=2, padding='same', use_bias=use_bias, kernel_regularizer=l2(1e-4),
                       bias_regularizer=l2(1e-4), name='conv1_conv'),
                BatchNormalization(epsilon=1.001e-5, name='conv1_bn'),
                ReLU(name='conv1_relu')
            ])

        self.pool1 = MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')

        self.blocks = Sequential()

        depth = base_depth

        for i, r in enumerate(layers):
            self.blocks.add(block(depth, strides=1 if i == 0 else 2, conv_shortcut=True, expansion=expansion,
                                  use_bias=use_bias,
                                  groups=groups, avg_down=avg_down, avd=avd,
                                  name='conv' + str(i + 2) + '_block1'))
            for j in range(2, r + 1):
                self.blocks.add(block(depth, expansion=expansion, avg_down=avg_down, avd=avd, use_bias=use_bias,
                                      groups=groups, name='conv' + str(i + 2) + '_block' + str(j)))
            depth *= 2

        if include_top:
            self.pool2 = GlobalAveragePooling2D(name='avg_pool')
            self.fc = Dense(num_classes, activation=classifier_activation, kernel_regularizer=l2(1e-4),
                            bias_regularizer=l2(1e-4), name='predictions')
        else:
            self.pool2 = tf.identity
            self.fc = tf.identity

    def call(self, inputs, **kwargs):
        outputs = self.stem(inputs)
        outputs = self.pool1(outputs)
        outputs = self.blocks(outputs)
        outputs = self.pool2(outputs)
        return self.fc(outputs)


class ResNet8(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet8, self).__init__([1, 1, 1], *args, block=BasicBlock, expansion=1, use_bias=False, **kwargs)


class ResNet14(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet14, self).__init__([2, 2, 2], *args, block=BasicBlock, expansion=1, use_bias=False, **kwargs)


class ResNet18(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__([2, 2, 2, 2], *args, block=BasicBlock, expansion=1, use_bias=False, **kwargs)


class ResNet34(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet34, self).__init__([3, 4, 6, 3], *args, block=BasicBlock, expansion=1, use_bias=False, **kwargs)


class ResNet26(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet26, self).__init__([2, 2, 2, 2], *args, **kwargs)


class ResNet50(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__([3, 4, 6, 3], *args, **kwargs)


class ResNet101(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet101, self).__init__([3, 4, 23, 3], *args, **kwargs)


class ResNet152(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet152, self).__init__([3, 8, 36, 3], *args, **kwargs)


class ResNeXt50(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNeXt50, self).__init__([3, 4, 6, 3], *args, expansion=2, use_bias=False, groups=32, base_depth=128, **kwargs)


class ResNeXt101(ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNeXt101, self).__init__([3, 4, 23, 3], *args, expansion=2, use_bias=False, groups=32, base_depth=128, **kwargs)
