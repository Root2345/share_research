import datetime
import time
import math
import soundfile as sf
import scipy.io.wavfile
import scipy.io
import scipy
from torch.autograd import Variable
import numpy as np
import os
import csv
import librosa
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sinctdnn import SincTDNN
from tqdm import tqdm
import my_data_loader as dl


class PretrainedSincTDNN(nn.Module):
    def __init__(self, model, classes=4471):
        super().__init__()
        self.sinctdnn = model
        self.output = nn.Linear(512, classes)
        # model.add_module('output', nn.Linear(512, classes))
    
    def forward(self, x):
        x = self.sinctdnn(x)
        x = self.output(x)
        return x


def ReadList(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig


def create_batches_rnd(batch_size, data_path, wlen):
    # 全部入りで抽出するので別にランダムではないです
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])

    # select a random sentence from the list  (joint distribution)
    [fs, signal] = scipy.io.wavfile.read(data_path)
    signal = signal.astype(float)/32768
    signal = librosa.util.normalize(signal)

    # accesing to a random chunk
    snt_len = signal.shape[0]
    # randint(0, snt_len-2*wlen-1)
    # snt_beg = np.random.randint(snt_len-wlen-1)
    snt_beg = 0
    snt_end = snt_beg+wlen

    if len(signal) < snt_end:
        signal = np.concatenate((signal, np.zeros(wlen - len(signal))))

    inp = torch.from_numpy(signal[snt_beg:snt_end]).float(
    ).contiguous().unsqueeze(dim=0).unsqueeze(dim=2)  # Current Frame

    return inp


def extract(model, target, inputs):
    feature = None

    def forward_hook(module, inputs, outputs):
        # 順伝搬の出力を features というグローバル変数に記録する
        global features
        # 1. detach でグラフから切り離す。
        # 2. clone() でテンソルを複製する。モデルのレイヤーで ReLU(inplace=True) のように
        #    inplace で行う層があると、値がその後のレイヤーで書き換えられてまい、
        #    指定した層の出力が取得できない可能性があるため、clone() が必要。
        features = outputs.detach().clone()

    # コールバック関数を登録する。
    handle = target.register_forward_hook(forward_hook)

    # 推論する
    model.eval()
    model(inputs)

    # コールバック関数を解除する。
    handle.remove()

    return features


def extractor(path_state_dict, path_save_npy, model_flag=0):
    model = SincTDNN()
    load_path = path_state_dict
    if model_flag == 0:
        classes = 155
        model.add_module('output', nn.Linear(512, classes))
        model.add_module('softmax', nn.Softmax(dim=1))

        load_weights = torch.load(load_path, map_location=torch.device('cpu'))[
            'model_state_dict']
        model.load_state_dict(load_weights)

    elif model_flag == 1:
        model = PretrainedSincTDNN(model)
        load_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/state_dict/agevc_xvec_124.model'
        load_weights = torch.load(load_path, map_location=torch.device('cpu'))[
            'model_state_dict']
        model.load_state_dict(load_weights)

    elif model_flag == 2:
        model = PretrainedSincTDNN(model, classes=155)
        # load_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/state_dict/agevc_agevec_466.model'
        load_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/state_dict/agevc_agevec_342.model'
        # load_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/state_dict/agevc_agevec_606.model'
        load_weights = torch.load(load_path, map_location=torch.device('cpu'))[
            'model_state_dict']
        model.load_state_dict(load_weights)
    else:
        load_path = '/home/s226059/workspace/git_space/workspace/embedding.pth'
        load_weights = torch.load(load_path, map_location=torch.device('cpu'))
        model.load_state_dict(load_weights)

    batch_size = 1
    # data_folder = '/home/s226059/workspace/Audio_data/16kHz/'
    # wav_lst = ReadList(
    #     '/home/s226059/workspace/git_space/workspace/self_sincnet/rakuten_scp.txt')
    datas = dl.get_datas(path='/home/s226059/workspace/split_utt_age_new.csv')
    wav_lst = [p[1] for p in datas]
    
    N_snt = len(wav_lst)
    fs = 16000
    duration = 4.0
    wlen = int(fs*duration)

    features = []

    print("Start Extracting X-vector")

    for i, data_path in enumerate(wav_lst):
        if (i + 1) % 500 == 0 or i + 1 == len(wav_lst):
            print("Extracting... ({}/{})".format(i + 1, len(wav_lst)))

        inp = create_batches_rnd(batch_size, data_path, wlen)
        target_module = model.sinctdnn.tdnn_.segment7
        # target_module = model.tdnn_.segment7
        features.append(np.array(extract(model, target_module, inp))[0])

    print("Extract Successed")

    # npy_path = '/home/s226059/workspace/git_space/workspace/self_sincnet/vector/np_stvec13.npy'
    np.save(path_save_npy, features)
    print("Saved to " + path_save_npy)
