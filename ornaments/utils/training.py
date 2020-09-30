import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.losses import LossFunctionWrapper


def optimize_learning_rate(model, train_ds, batch_size, loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
                           initial_learning_rate=1e-7, growth_rate=1.3, steps_per_epoch=10, epochs=55):
    tf.keras.backend.set_learning_phase(1)

    ds_size = batch_size*steps_per_epoch
    train_ds = train_ds.take(ds_size).shuffle(ds_size).batch(batch_size)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=steps_per_epoch,
                                                                 decay_rate=growth_rate, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = LossFunctionWrapper(loss_fn)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    loss_history = []
    lr_history = []

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss_value = loss(labels, predictions)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss_value)

    for epoch in range(epochs):
        for images, labels in train_ds:
            train_step(images, labels)

        lr_history.append(optimizer._decayed_lr(tf.float32))
        loss_history.append(train_loss.result())
        train_loss.reset_states()

        template = 'Epoch {}, Loss: {}, Learning Rate: {}'
        tf.print(template.format(epoch + 1, loss_history[-1], lr_history[-1]), output_stream=sys.stdout)

    loss_history = np.where(np.isnan(loss_history), np.inf, loss_history)
    derivatives = np.gradient(loss_history)
    max_decay = np.argmin(derivatives)

    return float(lr_history[max_decay])


def train_categorical_model(model, epochs, train_ds, val_ds, batch_size, optimizer=None, debug=False,
                            checkpoint_dir='checkpoints', load_weights=None, log_dir='logs/gradient_tape/',
                            save_best_only=False, loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
                            patience=None, accuracy_fn=tf.keras.metrics.sparse_categorical_accuracy):

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    loss = LossFunctionWrapper(loss_fn)
    accuracy = MeanMetricWrapper(accuracy_fn, name='accuracy')

    callbacks = []

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir.replace('/', os.sep).replace('\\', os.sep)
    log_dir = os.path.join(log_dir, current_time)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    checkpoint_dir = os.path.join(checkpoint_dir, current_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5'),
        save_weights_only=True, save_best_only=save_best_only))

    if patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True))

    model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy], run_eagerly=debug)

    if load_weights is not None:
        images, labels = next(iter(train_ds.batch(batch_size)))
        model(images)
        model.load_weights(load_weights)

    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
