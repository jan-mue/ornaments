from ornaments.modeling import LatticeModel
from ornaments.data_loading import load_data
from ornaments.wallpaper_groups import group_lattice_mapping


BATCH_SIZE = 32

if __name__ == '__main__':
    train_ds, test_ds = load_data(height=64, width=64)
    test_ds = test_ds.map(lambda img, label: (img, group_lattice_mapping[label]))

    test_ds = test_ds.batch(BATCH_SIZE)

    model = LatticeModel()
    images, labels = next(iter(train_ds.batch(BATCH_SIZE)))
    model(images)
    model.load_weights('models/lattice_model.h5')
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    loss, accuracy = model.evaluate(test_ds)

    print('Test Loss: {}, Test Accuracy: {}'.format(loss, accuracy * 100))
