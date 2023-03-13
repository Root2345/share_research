import sys, time, os, argparse, random
import importlib, itertools
import numpy
import glob
import datetime
import warnings
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from sinctdnn import SincTDNN
from XvecDataloader import *

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description = "Xvector")

parser.add_argument('--config', type=str, default=None, help='Config YAML file')

## Data loader
parser.add_argument('--max_sec', type=int, default=4, help='Input seconds of audio to the network for training')
parser.add_argument('--batch_size', type=int, default=96, help='Batch size, number of speaker per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500, help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of loader threads')
parser.add_argument('--augment', type=bool, default=False, help='Augment input')
parser.add_argument('--seed', type=int, default=123, help='Seed for the random number generator')

## Training details
parser.add_argument('--max_epoch', type=int, default=5000, help='Maximum number of epochs')
parser.add_argument('--trainfunc', type=str, default="", help='Loss function')

## Optimizer
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam')
parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',  type=float, default=0.001, help='Learning rate')
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
# parser.add_argument('--nClasses',       type=int,   default=4471,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--nClasses',       type=int,   default=155,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
# parser.add_argument('--save_path',      type=str,   default="/home/s226059/workspace/self_sinctdnn/save_model/agevc_xvec", help='Path for model and logs')
parser.add_argument('--save_path',      type=str,   default="/home/s226059/workspace/self_sinctdnn/save_model/agevc_agevec", help='Path for model and logs')

## Training and test data
# parser.add_argument('--train_list',     type=str,   default="/home/s226059/workspace/data/age_training/wav/utt2spk.train",  help='Train list')
parser.add_argument('--train_list',     type=str,   default="/home/s226059/workspace/data/age_training/wav/utt2aslab.train",  help='Train list')
parser.add_argument('--train_path',     type=str,   default="/home/s226059/workspace/data/age_training/wav/train", help='Absolute path to the train set')
parser.add_argument('--musan_path',     type=str,   default="/home/s226059/workspace/data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/home/s226059/workspace/data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):
        
        self.__model__ = speaker_model

        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.__scheduler__ = torch.optim.lr_scheduler.StepLR(self.__optimizer__, step_size=10, gamma=args.lr_decay)
        self.__lossfunc__ = nn.CrossEntropyLoss()
        self.lr_step = "epoch"

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]


    def train_network(self, loader, verbose):

        self.__model__.train()
        
        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        iter = 0

        tstart = time.time()
    
        for data, data_label in loader:
            data = data.transpose(2, 1).cuda(args.gpu).contiguous()

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda(args.gpu).contiguous()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                pout = (self.__model__(data))

                pred = torch.max(pout, dim=1)[1]
                cost = self.__lossfunc__
                nloss = cost(pout, label.long())
                err = torch.mean((pred != label.long()).float())

                # nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()
            
            loss += nloss.detach().cpu().item()
            # top1 += prec1.detach().cpu().item()
            top1 += err.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                sys.stdout.flush()
            
            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()
        
        return (loss / counter, top1 / counter)


    
    def saveParameters(self, path, epoch):

        # torch.save(self.__model__.module.state_dict(), path)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.__model__.state_dict(),
                    'optimizer_state_dict': self.__optimizer__.state_dict(),
                    'loss': self.__lossfunc__,},
                    path)
        
    
    def loadParameters(self, path):
        
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)


class PretrainedSincTDNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sinctdnn = model
        load_path = '/home/s226059/workspace/self_sinctdnn/save_model/embedding.pth'
        load_weights = torch.load(load_path)
        self.sinctdnn.load_state_dict(load_weights)
        classes = 155
        self.output = nn.Linear(512, classes)
        # model.add_module('output', nn.Linear(512, classes))
    
    def forward(self, x):
        x = self.sinctdnn(x)
        x = self.output(x)
        return x


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    model = SincTDNN()
    model_init = True

    if model_init:
        model = PretrainedSincTDNN(model)
        
    
    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    it = 1
    eer = [100]

    # if args.gpu == 0:
    scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    train_dataset = TrainDatasetLoader(**vars(args))
    train_sampler = TrainDatasetSampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer = ModelTrainer(model, **vars(args))

    for ii in range(1, it):
        trainer.__scheduler__.step()

    for it in range(it, args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        # if args.gpu == 0:
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
        scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))
        
        trainer.saveParameters(args.model_save_path+"/model%09d.model"%it, it)
        


    # if args.gpu == 0:
    scorefile.close()



def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""
    args.distributed = False
    args.gpu = "cuda:2"

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:', args.save_path)
    print('Use GPU: ', )

    # mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(args.gpu, 1, args)

if __name__ == '__main__':
    main()