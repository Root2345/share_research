import os
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
import my_data_loader
import pdb


EPOCHS = 500
BATCHSIZE = 16
CLASSES = 59
DATA_ARG = True
TEST_GROUP = 0


def set_tf_dataset(spectrograms, labels, classes, shuffle=True, batch_size=32):
    spectrograms = np.array(spectrograms)
    ds = tf.data.Dataset.from_tensor_slices((spectrograms, tf.one_hot(labels, classes)))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    return ds


def xvec_model(classes, input_shape):
    # モデルの構築
    model = models.Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Masking(mask_value=-10))
    model.add(layers.Conv1D(512, 5, padding='same', dilation_rate=5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(512, 5, padding='same', dilation_rate=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(512, 7, padding='same', dilation_rate=4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(512, 1, activation='relu'))
    model.add(layers.Conv1D(512, 1, activation='relu'))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(classes, activation="softmax"))

    met = CategoricalAccuracy(False)

    model.summary()

    # learning_rate = 1.1214526691368884e-08
    learning_rate = 1e-6

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[met])

    return model


def model_train(model, tr_ds, va_ds, ts_ds, epochs, log, model_name):
    # Clear clutter from previous TensorFlow graphs
    tf.keras.backend.clear_session()

    # TensorBoard callback
    tb = TensorBoard(log_dir=os.path.join('./logs', log),
                     histogram_freq=1,
                     write_images=True,
                     update_freq='batch',
                     )

    # Early Stopping
    es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=30)

    # Model Checkpoint
    mc = tf.keras.callbacks.ModelCheckpoint(filepath=model_name,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='min',
                                            period=1)

    # Optuna Callback

    histroy = model.fit(
        tr_ds,
        epochs=epochs,
        validation_data=va_ds,
        callbacks=[tb, mc]
    )

    eva = model.evaluate(ts_ds)
    predictions = model.predict(ts_ds)


# 中間層の出力用モデル
def get_hidden_layer(model, layer_name, data):
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)

    return intermediate_output


def main():
    age_th = 13
    test_group = TEST_GROUP
    classes = CLASSES
    epochs = EPOCHS
    batch_size = BATCHSIZE

    log = "test/xvec_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name = "model/model_xvec_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if DATA_ARG == True:
        spectrograms, datas = my_data_loader.set_spectrograms_and_datas()

    spectrograms = spectrograms.transpose(0, 2, 1)
    train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels = my_data_loader.load_datas(
        spectrograms, datas, age_th, test_group, classes)

    tr_ds = set_tf_dataset(train_specs, train_labels, classes=classes, batch_size=batch_size)
    va_ds = set_tf_dataset(valid_specs, valid_labels, classes=classes, shuffle=False, batch_size=1)
    ts_ds = set_tf_dataset(test_specs, test_labels, classes=classes, shuffle=False, batch_size=1)

    input_shape = spectrograms[0].shape
    print(input_shape)

    model = xvec_model(classes, input_shape)
    model_train(model, tr_ds, va_ds, ts_ds, epochs, log, model_name)

    # モデルの保存
    # model.save_weights(model_name)


# ----------------------For Optuna----------------------
def create_model(trial):

    # Hyperparameters to be tuned by Optuna.
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-5, log=True)

    input_shape = [100, 257]

    # Compose model
    model = tf.keras.models.Sequential()

    model.add(layers.Input(input_shape))
    model.add(layers.Masking(mask_value=-10))
    model.add(layers.Conv1D(512, 5, padding='same', dilation_rate=5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(512, 5, padding='same', dilation_rate=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(512, 7, padding='same', dilation_rate=4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(512, 1, activation='relu'))
    model.add(layers.Conv1D(512, 1, activation='relu'))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(CLASSES, activation="softmax"))

    # met = CategoricalAccuracy(False)
    loss = tf.keras.losses.CategoricalCrossentropy()

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=['accuracy'])

    return model


def objective(trial):
    # Clear clutter from previous TensorFlow graphs
    tf.keras.backend.clear_session()

    model = create_model(trial)
    model_name = "model/model_xvec_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = "test/xvec_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if DATA_ARG == True:
        spectrograms, datas = my_data_loader.set_spectrograms_and_datas()
    else:
        spectrograms, datas = my_data_loader.set_spectrograms_and_datas()

    spectrograms = spectrograms.transpose(0, 2, 1)
    train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels = my_data_loader.load_datas(
        spectrograms, datas, age_th=13, test_group=TEST_GROUP, classes=CLASSES)

    tr_ds = set_tf_dataset(train_specs, train_labels, classes=CLASSES, batch_size=BATCHSIZE)
    va_ds = set_tf_dataset(valid_specs, valid_labels, classes=CLASSES, shuffle=False, batch_size=1)
    ts_ds = set_tf_dataset(test_specs, test_labels, classes=CLASSES, shuffle=False, batch_size=1)

    # TensorBoard callback
    tb = TensorBoard(log_dir=os.path.join('./logs', log),
                     histogram_freq=1,
                     write_images=True,
                     update_freq='batch',
                     )

    # Early Stopping
    es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=50)

    # Model Checkpoint
    mc = tf.keras.callbacks.ModelCheckpoint(filepath=model_name,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='min',
                                            period=1)

    pr = TFKerasPruningCallback(trial, monitor='val_loss')

    history = model.fit(
        tr_ds,
        epochs=EPOCHS,
        validation_data=va_ds,
        callbacks=[tb, es, mc, pr]
    )

    return history.history['val_loss'][-1]


def show_result(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def _main():

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )

    study.optimize(objective, n_trials=25, timeout=600)

    show_result(study)


if __name__ == '__main__':
    main()
