import numpy as np
import csv
import librosa
import librosa.display
import random

random.seed(1234)

def get_datas(path='/home/s226059/workspace/split_utt_age_choice.csv'):
    '''
    split_utt_age.csvの内容を
    datas の中に格納して返す
    0: split group(0~9)
    1: filepath(str)
    2: utterance(str)
    3: sex(str)
    4: age(2~59)
    5: argment data or not(0, 1)
    6: データ数の制限(0, 1)
    # '''
    print("label file path:"+path)
    with open(path, 'r', encoding='utf-8-sig') as f:
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


def slice_spectrograms(spectrograms, datas):
    """
    スペクトログラムのスライス処理
    in : スペクトログラム・ファイル名・ラベル
    out: スライス後のスペクトログラム・ファイル名・ラベル
    """
    sl_spectrograms = []
    sl_datas = []

    for s in range(len(spectrograms)):
        # numpy に変換
        spectrogram = np.array(spectrograms[s])
        flames = spectrogram.shape[1]

        slice_frame = 100
        slide_flame = 50
        split_num = (flames - slice_frame) // slide_flame + 1
        if flames - slice_frame < 0:
            emb = slice_frame - flames
            padd = np.full((257, emb), 0)
            spectrogram = np.concatenate([spectrogram, padd], axis=1)

        for i in range(split_num):
            # スライスする要素の定義
            start_frame = slide_flame * i
            end_frame = start_frame + slice_frame
            
            # 配列のスライス
            spectrogram_slice = spectrogram[:, start_frame:end_frame]
            # Conv1dの実験のために一時的にコメントアウト
            # spectrogram_slice = np.expand_dims(spectrogram_slice, -1)
            sl_spectrograms.append(spectrogram_slice)
            sl_datas.append(datas[s])
    
    return sl_spectrograms, sl_datas


def set_spectrograms_and_datas():
    datas = get_datas()
    spectrograms = []

    for i in range(len(datas)):
        data = datas[i]
        # waveformを取得
        waveform = load_waveform(data[1])
        # spectrogramに変換
        spectrogram = np.array(get_spectrogram(waveform))
        spectrograms.append(spectrogram)
    
    # spectrogramを50フレームにスライス
    spectrograms, datas = slice_spectrograms(spectrograms, datas)

    spectrograms = np.array(spectrograms)
    print(spectrograms.shape)

    return spectrograms, datas


def get_mfcc(waveform):
    '''
    waveformを正規化済みのMFCCに変換する
    '''
    sr = 16000
    win_length = 512
    hop_length = win_length // 2
    n_fft = win_length
    n_mfcc = 23
    window = 'hann'
    melspec = librosa.feature.melspectrogram(y=waveform,
                                             sr=sr,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             win_length=win_length,
                                             window=window,
                                             center=True,
                                             pad_mode='reflect',
                                             power=2.0)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec),
                                sr=sr,
                                n_mfcc=n_mfcc,
                                dct_type=2,
                                norm='ortho',
                                lifter=0)
    mfcc = librosa.util.normalize(mfcc)
    return mfcc


def slice_mfccs(mfccs, datas):
    """
    MFCCのスライス処理
    in : MFCC・ファイル名・ラベル
    out: スライス後のMFCC・ファイル名・ラベル
    """
    sl_mfccs = []
    sl_datas = []

    for s in range(len(mfccs)):
        # numpy に変換
        mfcc = np.array(mfccs[s])
        flames = mfcc.shape[1]

        slice_frame = 400
        split_num = (flames - slice_frame) // 20 + 1
        if flames - slice_frame < 0:
            print("フレーム数が足りません")

        for i in range(split_num):
            # スライスする要素の定義
            start_frame = 20 * i
            end_frame = start_frame + slice_frame
            
            # 配列のスライス
            mfcc_slice = mfcc[:, start_frame:end_frame]
            mfcc_slice = np.expand_dims(mfcc_slice, -1)
            sl_mfccs.append(mfcc_slice)
            sl_datas.append(datas[s])
    
    return sl_mfccs, sl_datas


def set_mfccs_and_datas():
    datas = get_datas()
    mfccs = []

    for i in range(len(datas)):
        data = datas[i]
        # waveformを取得
        waveform = load_waveform(data[1])
        # mfccに変換
        mfcc = get_mfcc(waveform)
        mfccs.append(mfcc)
    
    # mfccを50フレームにスライス
    mfccs, datas = slice_mfccs(mfccs, datas)

    return mfccs, datas


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

    ageクラス
    年齢ラベル = 出力
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

    elif classes == 3:
        if age <= age_threshold:
            label = 0
        else:
            if sex == 'male':
                label = 1
            else:
                label = 2

    elif classes == 2:
        if age <= age_threshold:
            label = 0
        else:
            label = 1

    else:
        label = age

    return label


def rand_indexs_nodup(from_num, to_num):
    """
    データを指定するランダムなインデックスを生成
    from_num: 元データの個数
    to_num: 制限後データの個数(train:400, valid/test:50)
    """
    ns = []
    while len(ns) < to_num:
        n = random.randint(0, from_num-1)
        if not n in ns:
            ns.append(n)
    return ns


def load_datas(features, datas, age_th, test_group, classes=3):
    """
    features: 特徴量
    datas: ラベル・パス等
    age_th: 年齢閾値
    test_group: テストデータとして使用するデータのグループ番号
    class: 分類クラスの数
    """
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
        choice = int(data[6])

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
            if choice == 1:
                label = get_adch_label(data, classes, age_th)
                train_features.append(feature)
                train_labels.append(label)

    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)
    test_labels = np.array(test_labels)

    # データの絞り込み
    

    if classes == 3:
        print("train_data:", len(train_features))
        print("若年者:{0}, 大人(男性):{1}, 大人(女性):{2}".format(np.count_nonzero(train_labels == 0), np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == 2)))
        print("validation_data:",len(valid_features))
        print("若年者:{0}, 大人(男性):{1}, 大人(女性):{2}".format(np.count_nonzero(valid_labels == 0), np.count_nonzero(valid_labels == 1), np.count_nonzero(valid_labels == 2)))
        print("test_data:",len(test_features))
        print("若年者:{0}, 大人(男性):{1}, 大人(女性):{2}".format(np.count_nonzero(test_labels == 0), np.count_nonzero(test_labels == 1), np.count_nonzero(test_labels == 2)))
    elif classes == 2:
        print("train_data:", len(train_features))
        print("若年者:{0}, 大人:{1}".format(np.count_nonzero(train_labels == 0), np.count_nonzero(train_labels == 1)))
        print("validation_data:",len(valid_features))
        print("若年者:{0}, 大人:{1}".format(np.count_nonzero(valid_labels == 0), np.count_nonzero(valid_labels == 1)))
        print("test_data:",len(test_features))
        print("若年者:{0}, 大人:{1}".format(np.count_nonzero(test_labels == 0), np.count_nonzero(test_labels == 1)))
    else:
        print("train_data:", len(train_features))
        print("validation_data:",len(valid_features))
        print("test_data:",len(test_features))

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

