import os
import datetime
from librosa.core.audio import load
from librosa.core.spectrum import stft
import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import librosa
import librosa.display


def get_datas():
    '''
    split_utt_age.csvの内容を
    datas の中に格納して返す
    0: split group(0~9)
    1: filepath(str)
    2: utterance(str)
    3: sex(str)
    4: age(2~59)
    5: argment data or not(0, 1)
    '''
    with open('/home/s226059/workspace/split_utt_age.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        datas = [row for row in reader]

    return datas


def load_waveform(audio_file):
    '''
    filepathを入力にとり、librosaでwaveformに変換する
    '''
    waveform, _ = librosa.load(audio_file, sr=16000, mono=True)
    waveform = librosa.util.normalize(waveform)
    return waveform


def get_spectrogram(waveform):
    sr = 16000
    win_length = 512
    hop_length = win_length // 2
    n_fft = win_length
    window = 'hann'
    stft = librosa.stft(y=waveform,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window,
                        center=True)
    amplitude = np.abs(stft)
    spectrogram = librosa.amplitude_to_db(amplitude, ref=np.max)
    spectrogram = librosa.util.normalize(spectrogram)
    return spectrogram


def padding(spectrograms):
    length = []
    padd_specs = []

    # スペクトログラムの時間軸フレーム数を抽出
    for i in range(len(spectrograms)):
        length.append(spectrograms[i].shape)

    # フレームの最大値を取得
    # max_frame = max(length)[1]
    # 一個だけ長過ぎるデータがあるので1000フレームに設定
    max_frame = 500

    # スペクトログラムをパディング
    for sp in spectrograms:
        # フレーム数を取得
        frame = len(sp[1])
        # 最大フレームの音声でなければパディングを実行
        if frame <= max_frame:
            # 最大フレームとの差分
            emb = max_frame - frame
            # パディング用の配列を作成
            padd = np.full((257, emb), 1)
            # スペクトログラムとパディング用配列を結合
            padd_sp = np.concatenate([sp, padd], axis=1)
        # 最大フレームの音声ならばそのまま出力
        # 最大フレーム以上なら切り捨てる
        else:
            padd_sp = sp[:, :500]
        # パディング済みのスペクトログラムを出荷
        padd_specs.append(padd_sp)

    return padd_specs


def set_spectrograms_and_datas():
    datas = get_datas()
    spectrograms = []

    for i in range(len(datas)):
        data = datas[i]
        # waveformを取得
        waveform = load_waveform(data[1])
        # spectrogramに変換
        spectrogram = get_spectrogram(waveform)
        spectrograms.append(spectrogram)

    # パディング
    spectrograms = padding(spectrograms)
    # np.arrayに変換
    spectrograms = np.array(spectrograms)

    return spectrograms, datas


def get_adch_label(data, classes, age_threshold):
    '''
    4クラス
    若年者(男性) → 0, 若年者(女性) → 1, 
    大人(男性) → 2, 大人(女性) → 3

    3クラス
    若年者 → 0
    大人(男性) → 1, 大人(女性) → 2

    2クラス
    若年者 → 0
    大人 → 1
    '''
    age = int(data[4])
    sex = data[3]

    if classes == 4:
        if age <= age_threshold:
            if sex == 'male':
                label = 0
            else:
                label = 1
        else:
            if sex == 'male':
                label = 2
            else:
                label = 3

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


def load_datas(features, datas, age_th, test_group, classes):

    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []
    test_features = []
    test_labels = []

    for i in range(len(features)):
        feature = features[i]
        data = datas[i]
        group = int(data[0])
        arg = int(data[5])

        if group == test_group:
            if arg == 0:
                label = get_adch_label(data, classes, age_th)
                test_features.append(feature)
                test_labels.append(label)

        elif group == (test_group + 1) % 10:
            if arg == 0:
                label = get_adch_label(data, classes, age_th)
                valid_features.append(feature)
                valid_labels.append(label)

        else:
            label = get_adch_label(data, classes, age_th)
            train_features.append(feature)
            train_labels.append(label)

    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    test_labels = np.array(test_labels)

    if classes == 4:
        print("train_data:", len(train_features))
        print("若年者(男性):{0}, 若年者(女性):{1}, 大人(男性):{2}, 大人(女性):{3}".format(np.count_nonzero(train_labels == 0),
                                                                        np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == 2), np.count_nonzero(train_labels == 3)))
        print("validation_data:", len(valid_features))
        print("若年者(男性):{0}, 若年者(女性):{1}, 大人(男性):{2}, 大人(女性):{3}".format(np.count_nonzero(valid_labels == 0),
                                                                        np.count_nonzero(valid_labels == 1), np.count_nonzero(valid_labels == 2), np.count_nonzero(valid_labels == 3)))
        print("test_data:", len(test_features))
        print("若年者(男性):{0}, 若年者(女性):{1}, 大人(男性):{2}, 大人(女性):{3}".format(np.count_nonzero(test_labels == 0),
                                                                        np.count_nonzero(test_labels == 1), np.count_nonzero(test_labels == 2), np.count_nonzero(test_labels == 3)))
    elif classes == 3:
        print("train_data:", len(train_features))
        print("若年者:{0}, 大人(男性):{1}, 大人(女性):{2}".format(np.count_nonzero(train_labels == 0),
                                                       np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == 2)))
        print("validation_data:", len(valid_features))
        print("若年者:{0}, 大人(男性):{1}, 大人(女性):{2}".format(np.count_nonzero(valid_labels == 0),
                                                       np.count_nonzero(valid_labels == 1), np.count_nonzero(valid_labels == 2)))
        print("test_data:", len(test_features))
        print("若年者:{0}, 大人(男性):{1}, 大人(女性):{2}".format(np.count_nonzero(test_labels == 0),
                                                       np.count_nonzero(test_labels == 1), np.count_nonzero(test_labels == 2)))
    elif classes == 2:
        print("train_data:", len(train_features))
        print("若年者:{0}, 大人:{1}".format(np.count_nonzero(train_labels == 0), np.count_nonzero(train_labels == 1)))
        print("validation_data:", len(valid_features))
        print("若年者:{0}, 大人:{1}".format(np.count_nonzero(valid_labels == 0), np.count_nonzero(valid_labels == 1)))
        print("test_data:", len(test_features))
        print("若年者:{0}, 大人:{1}".format(np.count_nonzero(test_labels == 0), np.count_nonzero(test_labels == 1)))

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels
