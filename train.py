import os, datetime
import csv
import numpy as np
import tensorflow as tf
import torch
import random
import my_data_loader as dl
from extract_vector import extractor
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, Accuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow_addons.metrics import F1Score

random.seed(1234)
DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def encode_wav(wav_fn, model):
    """
    pytorchの学習済みモデルからx-vectorを取得
    """
    emb = model({"audio": wav_fn})
    return emb


def add_adch_label(data, age_threshold, classes):
    """
    データと年齢閾値を入力してラベルを取得
    """
    age = int(data[4])
    sex = data[3]

    if classes == 3:
        if age <= age_threshold:
            label = 0
        else:
            if sex == 'male':
                label = 1
            else:
                label = 2

    if classes == 2:
        if age <= age_threshold:
            label = 0
        else:
            label = 1

    return label


def read_xvecs(path):
    """
    npyファイルに保存したx-vectorを読み込み
    """
    return np.load(path)


def set_data_labels(xvecs, datas, test_group, age_th, classes):
    train_xvecs = []
    train_labels = []
    valid_xvecs = []
    valid_labels = []
    test_xvecs = []
    test_labels = []

    for i, feat in enumerate(xvecs):
        data = datas[i]
        group = int(data[0])
        arg = 0

        if group == test_group:
            if arg == 0:
                label = add_adch_label(data, age_th, classes)
                test_xvecs.append(feat)
                test_labels.append(label)
        
        elif group == (test_group + 1) % 10:
            if arg == 0:
                label = add_adch_label(data, age_th, classes)
                valid_xvecs.append(feat)
                valid_labels.append(label)
        
        else:
            if arg == 0:
                label = add_adch_label(data, age_th, classes)
                train_xvecs.append(feat)
                train_labels.append(label)

            # label = add_adch_label(data, age_th, classes)
            # train_xvecs.append(feat)
            # train_labels.append(label)

    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    test_labels = np.array(test_labels)
    train_xvecs = np.array(train_xvecs)
    valid_xvecs = np.array(valid_xvecs)
    test_xvecs = np.array(test_xvecs)

    return train_xvecs, train_labels, valid_xvecs, valid_labels, test_xvecs, test_labels


def get_xvecs():
    """
    モデルからx-vectorを抽出
    """

    # 学習済みモデル
    model = torch.hub.load('pyannote/pyannote-audio', "emb")

    # CSVファイルからwavファイルのパスのリストを取得
    path = '/home/s226059/workspace/split_utt_age_new.csv'
    datas = dl.get_datas(path)
    tr_datas = [list(tup) for tup in zip(*datas)]
    wav_path = tr_datas[1]

    # 埋め込みベクトルの抽出
    emb = []
    for i in wav_path:
        emb.append(np.array(encode_wav(i, model)))

    return emb


def set_tf_dataset(xvecs, labels, classes, shuffle=True, batch_size=32):
    """
    tensorflow 形式のデータセットオブジェクトを取得
    """
    xvecs = np.array(xvecs)
    ds = tf.data.Dataset.from_tensor_slices((xvecs, tf.one_hot(labels, classes)))
    if shuffle:
        ds = ds.shuffle(buffer_size=1200)
    ds = ds.batch(batch_size)
    return ds


def set_final(classes, batch_size, train_xvecs, train_labels, valid_xvecs, valid_labels, test_xvecs, test_labels):
    """
    訓練、評価、テストデータでそれぞれ処理
    """
    tr_ds = set_tf_dataset(train_xvecs, train_labels,
                           classes=classes, batch_size=batch_size)
    va_ds = set_tf_dataset(valid_xvecs, valid_labels,
                           classes=classes, shuffle=False, batch_size=1)
    ts_ds = set_tf_dataset(test_xvecs, test_labels,
                        classes=classes, shuffle=False, batch_size=1)

    return tr_ds, va_ds, ts_ds




def rand_indexs_nodup(from_num, to_num):
    """
    重複しない自然数を指定した範囲・個数だけ生成
    """
    ns = []
    while len(ns) < to_num:
        n = random.randint(0, from_num-1)
        if not n in ns:
            ns.append(n)
    return ns


def choice_datas(feature, labels, num=50):
    """
    対応した特徴量とラベルを指定した個数にランダム抽出
    feature: 特徴量
    labels: ラベル
    num: 絞り込む個数(train:400, valid/test:50)
    """
    labelby_feat = []
    use_labels = []

    for i in range(len(set(labels))):
        fe = feature[labels==i]
        rand_index = rand_indexs_nodup(from_num=len(fe), to_num=num)
        
        labelby_feat.append(fe[rand_index])
        use_labels.append(np.full(num, i))
        print(fe[rand_index].shape)

    use_feature = np.concatenate([labelby_feat[0], labelby_feat[1], labelby_feat[2]])
    use_labels = np.concatenate([use_labels[0], use_labels[1], use_labels[2]])

    return use_feature, use_labels


def data_process(classes, age_th, test_group, npy_path):
    """
    前処理の一括化関数
    """
    # x-vectorの読み込み(npyファイルから)
    xvecs = read_xvecs(npy_path)
    # CSVファイルの読み込み
    datas = dl.get_datas(path='/home/s226059/workspace/split_utt_age_new.csv')
    # 大人こどもラベルの設定
    # adch_labels = add_adch_label(datas, age_th)
    # 訓練・評価・テスト用で分割
    train_xvecs, train_labels, valid_xvecs, valid_labels, test_xvecs, test_labels = set_data_labels(xvecs, datas, test_group, age_th, classes)
    choice = True
    if choice == True:
        train_xvecs, train_labels = choice_datas(train_xvecs, train_labels, 400)
        valid_xvecs, valid_labels = choice_datas(valid_xvecs, valid_labels, 50)
        test_xvecs, test_labels = choice_datas(test_xvecs, test_labels, 50)
    # ニューラルネットワークに流し込む形式に変換
    tr_ds, va_ds, ts_ds = set_final(
        classes, 16, train_xvecs, train_labels, valid_xvecs, valid_labels, test_xvecs, test_labels)
    tr_ds_eva = set_tf_dataset(train_xvecs, train_labels,
                               classes=classes, batch_size=1)
    
    return tr_ds, va_ds, ts_ds, tr_ds_eva


def set_model(classes=2, learning_rate=0.00001):
    tf.keras.backend.clear_session()

    model = models.Sequential()

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    if classes == 2:
        model.add(layers.Dense(classes, activation="sigmoid"))
    elif classes > 2:
        model.add(layers.Dense(classes, activation="softmax"))

    # model.summary()
    if classes == 2:
        loss = tf.keras.losses.BinaryCrossentropy()
    elif classes > 2:
        loss = tf.keras.losses.CategoricalCrossentropy()

    # met = [tf.keras.metrics.CategoricalAccuracy(False), Precision(class_id=0), Recall(class_id=0), AUC(curve='ROC')]
    met = [tf.keras.metrics.CategoricalAccuracy(False), Precision(), Recall(), AUC(curve='ROC')]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=met)

    return model


def train(model, tr_ds, va_ds, log, epochs=30):
    tb = TensorBoard(log_dir=os.path.join('./logs', log),
                                  histogram_freq=1,
                                  write_images=True,
                                  update_freq='batch',
                                  )

    # Early Stopping
    es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=10)

    # Model Checkpoint
    # mc = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
    #                                         monitor='val_loss',
    #                                         verbose=1,
    #                                         save_best_only=True,
    #                                         save_weights_only=False,
    #                                         mode='min',
    #                                         period=1)

    histroy = model.fit(
        tr_ds,
        epochs=epochs,
        validation_data=va_ds,
        callbacks=[tb],
        verbose=2
    )

    return model


def get_evaluate(model, ds, age_th, test_group):
    """
    評価指標の取得とf値の計算
    """
    # データセットを入力してモデルを評価
    eva = model.evaluate(ds, verbose=0)
    precision = eva[2]
    recall = eva[3]
    # F値を計算して配列の最後に追加
    try:
        eva.append(2 * precision * recall / (precision + recall))
    except ZeroDivisionError:
        eva.append(0)
    

    head = ["ageth: " + str(age_th), 'group: ' + str(test_group)]
    head.extend(eva)
    print("loss: {} - categorical_accuracy: {} - precision: {} - recall: {} - auc: {} - fmeasure: {}".format(eva[0], eva[1], eva[2], eva[3], eva[4], eva[5]))
    return head


def valid_log(log_root, filename, write_data):
    """
    logファイルに記述
    """
    with open(os.path.join(log_root, filename), mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)


def cross_valid(classes, epochs, learning_rate, npy_path, save_type):
    print("params - classes: {}, epoch: {}, learing rate: {}".format(classes, epochs, learning_rate))
    # log_root = os.path.join('/home/s226059/workspace/git_space/workspace/self_sincnet/logs/stvec', DATETIME)
    log_root = os.path.join('/home/s226059/workspace/git_space/workspace/self_sincnet/logs/' + save_type, DATETIME)
    # model_save_root = os.path.join('/home/s226059/workspace/git_space/workspace/self_sincnet/save_model/xvec', DATETIME)
    model_save_root = os.path.join('/home/s226059/workspace/git_space/workspace/self_sincnet/save_model/' + save_type, DATETIME)
    os.makedirs(model_save_root)
    
    test_result = []
    # test_ageth = [18]
    # test_tgroup = [2]
    for age_th in range(9, 18):
    # for age_th in test_ageth:
        csv_header = ["age_th: " + str(age_th), "test group", "Loss", "Accuracy", "Precision", "Recall", "AUC", "F1Score"]
        
        tr_eva_sum = []
        va_eva_sum = []
        ts_eva_sum = []
        
        for test_group in range(0, 10):
        # for test_group in test_tgroup:
            print("年齢閾値: {}, テストグループ: {}".format(age_th, test_group))

            # ログ、モデル保存用のパスを定義
            add_path = 'ageth' + str(age_th) + '/test' + str(test_group)
            log = os.path.join(log_root, add_path)
            model_save_path = os.path.join(model_save_root, add_path)
            os.makedirs(model_save_path)
            model_save_path = model_save_path + '/model.h5'

            # データセットの設定
            tr_ds, va_ds, ts_ds, tr_ds_eva = data_process(classes, age_th, test_group, npy_path)
            # モデルの定義
            model = set_model(classes, learning_rate)
            # 学習
            model = train(model, tr_ds, va_ds, log, epochs)
            
            # モデルの保存
            model.save(model_save_path)

            # ログの記述
            if test_group == 0:
                valid_log(log_root, "train_val.csv", csv_header)
                valid_log(log_root, "valid_val.csv", csv_header)
                valid_log(log_root, "test_val.csv", csv_header)
            print("train data evaluate")
            tr_eva = get_evaluate(model, tr_ds_eva, age_th, test_group)
            print("validation data evaluate")
            va_eva = get_evaluate(model, va_ds, age_th, test_group)
            print("test data evaluate")
            ts_eva = get_evaluate(model, ts_ds, age_th, test_group)
            valid_log(log_root, "train_val.csv", tr_eva)
            valid_log(log_root, "valid_val.csv", va_eva)
            valid_log(log_root, "test_val.csv", ts_eva)
            tr_eva_sum.append(tr_eva[2:len(tr_eva)])
            va_eva_sum.append(va_eva[2:len(va_eva)])
            ts_eva_sum.append(ts_eva[2:len(ts_eva)])

        age_results = [tr_eva_sum, va_eva_sum, ts_eva_sum]
        stages = ["train", "valid", "test"]

        for result, stage in zip(age_results, stages):
            head = ["ageth: " + str(age_th), 'group: average']
            ave = np.mean(result, axis=0)
            head.extend(ave)
            valid_log(log_root, stage + "_val.csv", head)

            if stage == "test":
                test_result.append(head)

    for r in test_result:
        valid_log(log_root, "test_val.csv", r)


def main():
    # パラメータ設定
    extract = 0
    model_flag = 2 # 0: age-vec 1: x-vec
    # npy_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/vector/uninum_agevec.npy'
    npy_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/vector/uninum_re_test.npy'
    # npy_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/vector/uninum_re2_agevec.npy'
    # npy_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/vector/uninum_re3_agevec.npy'

    if extract == 1:
        path_state_dict = '/home/s226059/workspace/git_space/workspace/self_sincnet/state_dict/agevc_xvec_124.model'
        extractor(path_state_dict, npy_path, model_flag)
    
    classes = 3
    epochs = 50
    lr = [1e-5]
    save_type = "xvec_agevc"
    for i in lr:
        cross_valid(classes, epochs, i, npy_path, save_type)
    
    # save_type = "agevec"
    # npy_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/vector/uninum_re_agevec.npy'
    # for i in lr:
    #     cross_valid(classes, epochs, i, npy_path, save_type)

if __name__ == '__main__':
    main()