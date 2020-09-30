import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization

from ornaments.modeling import ResNet8, Correlation2D
from ornaments.data_loading import load_data
from ornaments.utils import train_categorical_model


BATCH_SIZE = 32
TRAIN_SIZE = 8946
TRAIN_VAL_SPLIT = 0.8
EPOCHS = 100


if __name__ == '__main__':
    train_ds, test_ds = load_data(height=128, width=128)

    val_ds = train_ds.take(TRAIN_SIZE).skip(int(TRAIN_SIZE * TRAIN_VAL_SPLIT))
    train_ds = train_ds.take(int(TRAIN_SIZE * TRAIN_VAL_SPLIT)).shuffle(10000)

    model = Sequential([
        Correlation2D(),
        BatchNormalization(epsilon=1.001e-5),
        ResNet8(17)
    ])

    images, labels = next(iter(train_ds.batch(BATCH_SIZE)))
    model(images)
    model.load_weights('models/ornaments_model.h5')

    bn = model.layers[1]
    resnet = model.layers[-1]

    bn.trainable = False
    resnet.stem.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    train_categorical_model(model, EPOCHS, train_ds, val_ds, BATCH_SIZE, optimizer, patience=10)

    model.save('models/ornaments_model/')
