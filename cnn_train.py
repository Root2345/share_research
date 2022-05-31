import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import wave
import csv

from tensorflow.python.ops.gen_linalg_ops import batch_cholesky
import audio_edit
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

import pdb

def process_dataset(audio_datas, labels):
    files_ds = tf.data.Dataset.from_tensor_slices(audio_datas)
    output_ds = (audio_datas, labels)

    return output_ds

def cnn_model_spect(age_threshold, split_group):
    train_mfccs, train_labels, test_mfccs, test_labels = audio_edit.get_datas(age_threshold, split_group, norm=True)

    # モデルの構築
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(23, 23, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', ))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', ))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='sigmoid'))

    met = CategoricalAccuracy(False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[met])

    td = tf.data.Dataset.from_tensor_slices(test_mfccs)
    tl = tf.data.Dataset.from_tensor_slices(tf.one_hot(test_labels, 4))
    tds = tf.data.Dataset.zip((td,tl)).shuffle(379).batch(1)

    early_stopping = EarlyStopping(patience=10, verbose=1) 
    
    # 学習
    history = model.fit(train_mfccs, tf.one_hot(train_labels, 4), 
                        batch_size=16, epochs=50, validation_data=tds, 
                        callbacks=[tf.keras.callbacks.TensorBoard('./logs/cnn', 1), early_stopping])
    
    # 評価
    loss, acc = model.evaluate(test_mfccs, tf.one_hot(test_labels, 4))

def cnn_model(train_mfccs, train_labels, test_mfccs, test_labels, classes, log, epochs, batch_size):
    # モデルの構築
    input_shape = train_mfccs.shape
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(23, 100, 1)))
    model.add(layers.Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(layers.Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(classes, activation='softmax'))

    met = CategoricalAccuracy(False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[met])

    
    # validation dataのセットアップ
    tr_d = tf.data.Dataset.from_tensor_slices(train_mfccs)
    # tr_l = tf.data.Dataset.from_tensor_slices(tf.one_hot(train_labels, classes))
    tr_l = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
    # tds = tf.data.Dataset.zip((td,tl)).shuffle(379).batch(1)

    def mapping_func(mfccs, labels):
        return mfccs, labels
    # tr_ds = tf.data.Dataset.zip((tr_d,tr_l))
    train_mfccs = tf.expand_dims(train_mfccs, -1)
    # train_mfccs = train_mfccs.reshape((-1, 23, 100, 1))
    test_mfccs = tf.expand_dims(test_mfccs, -1)
    # test_mfccs = test_mfccs.reshape((-1, 23, 100, 1))
    tr_ds = tf.data.Dataset.from_tensor_slices((train_mfccs, tf.one_hot(train_labels, classes)))

    te_d = tf.data.Dataset.from_tensor_slices(test_mfccs)
    te_l = tf.data.Dataset.from_tensor_slices(tf.one_hot(test_labels, classes))
    # tds = tf.data.Dataset.zip((td,tl)).shuffle(379).batch(1)
    # te_ds = tf.data.Dataset.zip((tr_d,tr_l))

    va_d = tf.data.Dataset.from_tensor_slices(test_mfccs)
    va_l = tf.data.Dataset.from_tensor_slices(tf.one_hot(test_labels, classes))
    # tds = tf.data.Dataset.zip((td,tl)).shuffle(379).batch(1)
    # va_ds = tf.data.Dataset.zip((va_d,va_l)).batch(batch_size)
    va_ds = tf.data.Dataset.from_tensor_slices((test_mfccs, tf.one_hot(test_labels, classes)))
    

    # early_stopping = EarlyStopping(patience=10, verbose=1)
    
    # 学習
    histroy = model.fit(
        tr_ds,
        epochs=epochs,
        validation_data=va_ds,
        callbacks=[tf.keras.callbacks.TensorBoard('./logs/{}'.format(log), 1)]
    )
    # history = model.fit(train_mfccs, tf.one_hot(train_labels, classes), 
    #                     batch_size=batch_size, epochs=epochs, validation_data=(test_mfccs, test_labels), 
    #                     callbacks=[tf.keras.callbacks.TensorBoard('./logs/{}'.format(log), 1)])

    # history = model.fit(train_mfccs, tf.one_hot(train_labels, 4), 
    #                     batch_size=16, epochs=50, validation_data=tds, 
    #                     callbacks=[tf.keras.callbacks.TensorBoard('./logs/cnn_test', 1), early_stopping])
    
    # 評価
    loss, acc = model.evaluate(test_mfccs, tf.one_hot(test_labels, classes))

def main():
    # for i in range(8, 19):
    #     for j in range(10):
    #         cnn_model(i, j)

    # データ準備
    age_threshold = 13
    split_group = 0
    train_mfccs, train_labels, test_mfccs, test_labels = audio_edit.get_datas(age_threshold, split_group, norm=True)

    # モデル学習
    dt_now = datetime.datetime.now()
    classes = 4
    log = "test/norm_on " + str(dt_now)
    epochs = 2
    batch_size = 16
    cnn_model(train_mfccs, 
              train_labels, 
              test_mfccs, 
              test_labels, 
              classes=classes, 
              log=log, 
              epochs=epochs, 
              batch_size=batch_size)

if __name__ == '__main__':
    main()
    