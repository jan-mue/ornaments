import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization

from ornaments.modeling import ResNet8, Correlation2D
from ornaments.data_loading import load_generated_data
from ornaments.utils import train_categorical_model


BATCH_SIZE = 64
TRAIN_SIZE = 614_482
TRAIN_VAL_SPLIT = 0.8
EPOCHS = 30


if __name__ == '__main__':
    train_ds, test_ds = load_generated_data(height=128, width=128)

    train_ds = train_ds.take(TRAIN_SIZE).shuffle(20000)
    val_ds = test_ds.take(int((1-TRAIN_VAL_SPLIT) * TRAIN_SIZE))

    model = Sequential([
        Correlation2D(),
        BatchNormalization(epsilon=1.001e-5),
        ResNet8(17)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    train_categorical_model(model, EPOCHS, train_ds, val_ds, BATCH_SIZE, optimizer, patience=3)

    model.save_weights('models/ornaments_model.h5')
