import tensorflow as tf

from ornaments.modeling import LatticeModel
from ornaments.data_loading import load_data
from ornaments.utils import train_categorical_model
from ornaments.wallpaper_groups import group_lattice_mapping


BATCH_SIZE = 256
TRAIN_SIZE = 8946
TRAIN_VAL_SPLIT = 0.8
EPOCHS = 300

if __name__ == '__main__':

    train_ds, test_ds = load_data(height=64, width=64)
    train_ds = train_ds.map(lambda img, label: (img, group_lattice_mapping[label]))

    val_ds = train_ds.take(TRAIN_SIZE).skip(int(TRAIN_SIZE * TRAIN_VAL_SPLIT))
    train_ds = train_ds.take(int(TRAIN_SIZE * TRAIN_VAL_SPLIT)).shuffle(10000)

    model = LatticeModel()

    initial_learning_rate = 5e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=50,
        decay_rate=0.95,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_categorical_model(model, EPOCHS, train_ds, val_ds, BATCH_SIZE, optimizer, patience=15)

    model.save_weights('models/lattice_model.h5')
