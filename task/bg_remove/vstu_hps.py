import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from copy import deepcopy

from torchvision.transforms import Compose

from model_bank.video_swin_transformer import SwinTransformer3D_unet
from loss import Lloss
from dataset.bg_remove import crop_3D_dataset, RandomHorizontalFlipVideo, PairedTransform

import numpy as np
from utils import save_yaml
from skimage import io
from utils import psnr, corr

from framework.common_callback import EpochCheckpoint, EpochTrainLog, EpochEvalLog
from accelerate.utils import set_seed
from framework.model import Model
import framework.hyperSearch as HPS

# ############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of training epochs")

parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--img_s', type=int, default=64, help="the slices of image sequence")
parser.add_argument('--img_w', type=int, default=128, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=128, help="the height of image sequence")
parser.add_argument('--gap_s', type=int, default=32, help='the slices of image gap')
parser.add_argument('--gap_w', type=int, default=64, help='the width of image gap')
parser.add_argument('--gap_h', type=int, default=64, help='the height of image gap')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam: weight_decay")

parser.add_argument('--normalize_factor', type=int, default=65535, help='normalize factor, val = 2**16')

parser.add_argument('--log_folder', type=str, default='/home/wwh/code/DLT-main/log/bg_remove/vstu', help="output directory")
parser.add_argument('--datasets_path', type=str, default='/home/wwh/Dataset/simulation/', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='remove_bessel_bg_signal', help="A folder containing files for training")
parser.add_argument('--remove_folder', type=list, default=[])

parser.add_argument('--select_img_num', type=int, default=4000, help='select the number of images')
parser.add_argument('--train_datasets_size', type=int, default=4000, help='datasets size for training')

parser.add_argument('--checkpoint', type=str, default='')
opt = parser.parse_args()

''' <<<<<<<<<<<<<<<<<<<< setup >>>>>>>>>>>>>>>>> '''
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


'''<<<<<<<<<<<<<<<<<<< dataset >>>>>>>>>>>>>>>>>>>>'''
transforms = Compose([
    RandomHorizontalFlipVideo(0.5)
])
paired_transform = PairedTransform(transforms)

train_set = crop_3D_dataset(opt, train=True, transform=paired_transform)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=16)
val_set = crop_3D_dataset(opt, train=False, transform=paired_transform)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

'''<<<<<<<<<<<<<<<<<<< loss function >>>>>>>>>>>>>>>>>>>>'''
hps_loss = HPS.ModuleSpace(Lloss, {"index": [0.3, 0.4, 0.5, 0.6, 0.7]})

'''<<<<<<<<<<<<<<<<<<< network >>>>>>>>>>>>>>>>>>>>'''
hps_network = HPS.ModuleSpace(SwinTransformer3D_unet, {"embed_dim": [8, 16, 32],
                                                       "depths": [[2, 6, 6, 2], [2, 2, 6, 2]],
                                                       "num_heads": [[2, 4, 8, 8], [2, 4, 8, 16]],
                                                       "drop_path_rate": [0.1, 0.2]})

''' <<<<<<<<<<< optimizer and lr_scheduler >>>>>>>>>>> '''
hps_seed = np.random.randint(0, 100, 5).tolist()
hps_optim = HPS.OptimSpace(["AdamW"], {"lr": [1e-4, 4e-5]})

hps_lr_scheduler = HPS.LrSchedulerSpace(["OneCycleLR"],
                                        {"max_lr": [HPS.HpsMul(10, "lr")], "total_steps": [HPS.HpsAdd(0, "total_steps")]})

''' <<<<<<<<<<< metrics >>>>>>>>>>> '''
metrics = {"psnr": psnr}

''' <<<<<<<<<<< callback >>>>>>>>>>> '''

hps_task = HPS.HyperParamSearch(
        network=hps_network,
        loss_fn=hps_loss,
        optimizer=hps_optim,
        train_dataset=train_loader,
        val_dataset=val_loader,
        epoch_space=[opt.n_epochs],
        lr_scheduler=hps_lr_scheduler,
        metrics_dict=metrics,
        seed_space=hps_seed,
        log_folder=opt.log_folder)

study_name = "hps"
from framework.hps_callback import EpochEvalLog, EpochTrainLog, EpochCheckpoint, Pruning
save_folder = os.path.join(opt.log_folder, study_name)
callbacks = [
    Pruning("val_psnr"),
    EpochCheckpoint(os.path.join(save_folder, "weights"), save_freq=1),
    EpochTrainLog(save_folder, save_freq=1),
    EpochEvalLog(save_folder, save_freq=1),
]
hps_task.fit(study_name, callbacks, monitor="val_psnr", direction="maximize", n_trials=100, pruner=None)





