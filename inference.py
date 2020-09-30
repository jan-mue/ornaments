import tensorflow as tf
from ornaments.data_loading import load_data


BATCH_SIZE = 8

if __name__ == '__main__':
    train_ds, test_ds = load_data(height=128, width=128)
    test_ds = test_ds.batch(BATCH_SIZE)

    model = tf.keras.models.load_model('models/ornaments_model/', compile=False)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    loss, accuracy = model.evaluate(test_ds)

    print('Test Loss: {}, Test Accuracy: {}'.format(loss, accuracy * 100))
