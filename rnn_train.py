from math import e
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.ops.gen_linalg_ops import batch_cholesky
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
import rnn_data_loader
from evaluate_index import FMeasure


def set_tf_dataset(spectrograms, labels, classes, shuffle=True, batch_size=32):
    spectrograms = np.array(spectrograms)
    ds = tf.data.Dataset.from_tensor_slices((spectrograms, tf.one_hot(labels, classes)))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    return ds



def rnn_model(classes):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=1, input_shape=(500, 257)))
    model.add(layers.SimpleRNN(128))
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()
    accuracy = CategoricalAccuracy(False)
    recall = Recall()
    precision = Precision()
    f_measure = FMeasure()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[accuracy, recall, precision, f_measure])

    return model


def lstm_model(classes):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=1, input_shape=(500, 257)))
    model.add(layers.LSTM(32))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(32))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(32))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()
    accuracy = CategoricalAccuracy(False)
    recall = Recall()
    precision = Precision()
    f_measure = FMeasure()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[accuracy, recall, precision, f_measure])

    return model


def gru_model(classes):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=1, input_shape=(500, 257)))
    model.add(layers.GRU(32))
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()
    accuracy = CategoricalAccuracy(False)
    recall = Recall()
    precision = Precision()
    f_measure = FMeasure()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[accuracy, recall, precision, f_measure])

    return model


def crnn_model(classes):
    model = models.Sequential()

    model.add(layers)

def model_train(model, tr_ds, va_ds, log, epochs):
    es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=5)

    histroy = model.fit(
        tr_ds,
        epochs=epochs,
        validation_data=va_ds,
        callbacks=[tf.keras.callbacks.TensorBoard('/home/s226059/workspace/logs/{}'.format(log), 1)]
    )

    return histroy


def evaluate(model, ts_ds):
    eva = model.evaluate(ts_ds)
    predictions = model.predict(ts_ds)
    
    return eva, predictions


def data_split(classes, batch_size, train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels):
    # 訓練データ・ラベルのテンソルを作成
    print("訓練データ作成開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tr_ds = set_tf_dataset(train_specs, train_labels, classes=classes, batch_size=batch_size)
    # 評価データ・ラベルのテンソルを作成
    print("評価データ作成開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    va_ds = set_tf_dataset(valid_specs, valid_labels, classes=classes, shuffle=False, batch_size=batch_size)
    # テストデータ・ラベルのテンソルを作成
    print("テストデータ作成開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ts_ds = set_tf_dataset(test_specs, test_labels, classes=classes, shuffle=False, batch_size=batch_size)

    return tr_ds, va_ds, ts_ds


# 年齢閾値別10分割交差検証
def closs_validation(spectrograms, datas, classes, batch_size, epochs):
    for i in range(11):
        age_th = 8 + i
        log_root = "rnn/ageth_{:02}".format(age_th)
        for test_group in range(10):
            log_dir = os.path.join(log_root, "test_group{}".format(test_group))

            train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels = rnn_data_loader.load_datas(spectrograms, datas, age_th, test_group, classes)
            
            tr_ds, va_ds, ts_ds = data_split(classes, batch_size, train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels)

            model = rnn_model(classes)
            histry = model_train(model, tr_ds, va_ds, log_dir, epochs)
            eva, predictions = (model, ts_ds)

def main():
    age_th = 13
    test_group = 0
    classes = 3
    dt_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = "test/rnn/" + str(dt_now)
    epochs = 20
    batch_size = 10
    spectrograms, datas = rnn_data_loader.set_spectrograms_and_datas()
    
    # npyファイルに書き出す
    # np.save('data/temp/spectrograms',spectrograms)
    # print(spectrograms.shape)
    
    # npyファイルからスペクトログラムを読み込む
    # print("スペクトログラムの読み込み開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # spectrograms = np.load('/home/s226059/workspace/git_space/workspace/data/temp/spectrograms.npy')
    # print("変換前のShape", spectrograms.shape)
    # ネットワークの次元数に合うように転置
    spectrograms = spectrograms.transpose(0, 2, 1)
    # print("スペクトログラムの読み込み完了" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("変換後のShape", spectrograms.shape)

    # ラベルファイルの読み込み
    # datas = rnn_data_loader.get_datas()
    
    # 訓練・テスト・評価データに分割
    train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels = rnn_data_loader.load_datas(spectrograms, datas, age_th, test_group, classes)
    
    # 訓練データ・ラベルのテンソルを作成
    print("訓練データ作成開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tr_ds = set_tf_dataset(train_specs, train_labels, classes=classes, batch_size=batch_size)
    # 評価データ・ラベルのテンソルを作成
    print("評価データ作成開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    va_ds = set_tf_dataset(valid_specs, valid_labels, classes=classes, shuffle=False, batch_size=batch_size)
    # テストデータ・ラベルのテンソルを作成
    print("テストデータ作成開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ts_ds = set_tf_dataset(test_specs, test_labels, classes=classes, shuffle=False, batch_size=batch_size)

    # モデルの選択
    print("モデルの定義")
    model = rnn_model(classes)
    # model = lstm_model(classes)
    # model = gru_model(classes)

    # 訓練
    print("訓練開始" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    history = model_train(model, tr_ds, va_ds, log, epochs)
    print("テストデータで評価" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    eva, predict = evaluate(model, ts_ds)



if __name__ == '__main__':
    main()