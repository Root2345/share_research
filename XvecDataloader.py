# Reference
# https://github.com/clovaai/voxceleb_trainer/blob/343af8bc9b325a05bcf28772431b83f5b8817f5a/DatasetLoader.py#L191
import torch
import numpy as np
import random
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# 指定した数で割ったあまりをもとの数から引く
def round_down(num, divisor):
    return num - (num%divisor)

# worker id をもとにシード値を設定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# wavファイルを読み込み
def loadWAV(filename, max_sec, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = 16000 * max_sec

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    # 4秒に満たない音をパディング
    if audiosize < max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    
    # 音声の開始位置の設定
    # 学習時：4秒間を確保できる位置からランダムに開始
    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    
    # 実際に4秒切り出す
    feats = []
    if evalmode and max_sec == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat


class AugmentWAV(object):
    """
    データ拡張用クラス
    """
    def __init__(self, musan_path, rir_path, max_sec):
        self.max_sec = max_sec
        self.max_audio = 16000 * max_sec

        self.noisetypes = ['noise', 'speech', 'music']
        
        self.noisesnr = {'noise':[0, 15], 'speech':[13, 20], 'music':[5, 15]}
        self.numnoise = {'noise':[1, 1], 'speech':[3, 7], 'music':[1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_file = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        # 元データのデシベルを計算する
        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            
            # 拡張に使用するノイズを読み込む
            noiseaudio = loadWAV(noise, self.max_sec, evalmode=False)
            # SNRをランダムで設定
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            # ノイズのデシベルを計算する
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4)
            # 元データのデシベル・ノイズのデシベル・SNRから実際に拡張に使用できるようパワーを調整してリストに追加
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))) * noiseaudio

        # 元音声と処理したノイズを足し合わせて返す
        return np.sum(np.concatenate(noise, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)

        rir, fs = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))

        # 読み込んだrirを用いて畳み込み（残響を付加）
        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]

class TrainDatasetLoader(Dataset):
    """
    学習時に使用するデータローダクラス
    train_list: 「学習データの話者ID・ルートディレクトリからのパスを記述したファイル」のパス
    augment: データ拡張を行うかどうか(True or False)
    musan_path: musan noiseのパス
    rir_path: rirs noiseのパス
    max_sec: 入力音声の最大秒数
    train_path: 学習データのルートディレクトリまでのパス
    """
    def __init__(self, train_list, augment, musan_path, rir_path, max_sec, train_path, **kwargs):
        
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_sec=max_sec)

        self.train_list = train_list
        self.max_sec = max_sec
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment

        # 学習用のファイルを読み込む
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # 話者IDと番号(出力ラベル)の辞書を作成
        # 話者IDの配列
        dictkeys = list(set([x.split()[1] for x in lines]))
        # 配列をソート
        dictkeys.sort()
        # 先頭から順番に番号を設定した辞書を作成
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # 学習データとラベルの対応関係を設定
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            # データのリスト
            data = line.strip().split()

            # 話者ID
            speaker_label = dictkeys[data[1]]
            # ファイルパス
            filename = os.path.join(train_path, data[0])

            # それぞれ追加
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):
        
        feat = []

        for index in indices:
            # 音声データを読み込む
            audio = loadWAV(self.data_list[index], self.max_sec, evalmode=False)

            # データ拡張の方法をランダムで決める
            if self.augment:
                augtype = random.randint(0, 4)
                if augtype == 1:
                    # 残響重畳のみ
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    # 音楽を付加
                    audio = self.augment_wav.additive_noise('music',audio)
                elif augtype == 3:
                    # バブルノイズ(群衆のざわめき)を付加
                    audio = self.augment_wav.additive_noise('speech',audio)
                elif augtype == 4:
                    # ノイズを付加
                    audio = self.augment_wav.additive_noise('noise',audio)

            feat.append(audio)

        feat = np.concatenate(feat, axis=0)
        
        # Torchのテンソル型に変換した特徴量とラベルを返す
        return torch.FloatTensor(feat), self.data_label[index]
        
    def __len__(self):
        return len(self.data_list)

# テストデータのデータローダは飛ばします

class TrainDatasetSampler(torch.utils.data.Sampler):
    """
    データセットの読み込むためのクラス
    data_source
    nPerSpeaker
    max_seg_per_spk
    batch_size
    distributed
    seed
    """
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):
        
        self.data_label = data_source.data_label
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed

    def __iter__(self):

        g = torch.Generator()
        # シード値の設定
        g.manual_seed(self.seed + self.epoch)
        # ラベルの数の整数の純烈のリストの生成(は？)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # 各IDに対応した番号の辞書型配列にソート
        for index in indices:
            # 話者IDの指定
            speaker_label = self.data_label[index]
            # 話者IDがデータの辞書になかったとき
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            # データ辞書に追加
            data_dict[speaker_label].append(index)
        
        # ファイルを各クラスにグループ化
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        # lst:リスト, sz:飛ばし数 →  リストを飛ばし数ごとに分割する
        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            # 指定した話者数の倍数にデータ数を丸め込み
            numSeg = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        # 学習データをシャッフルする
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []
        
        # 同じバッチ内の中に同じ話者のペアが2組入るのを防ぐ
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        # データをGPUに割り当てる
        if self.distributed:
            total_size = round_down(len(mixed_list), self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
