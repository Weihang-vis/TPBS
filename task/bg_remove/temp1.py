import os
import torch
import pickle
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from copy import deepcopy

from torchvision.transforms import Compose

from model_bank.network import Network_Uformer_3D_Unet
from loss import MyLoss
from dataset.bg_remove import crop_3D_dataset, RandomHorizontalFlipVideo, PairedTransform

import numpy as np
from utils import save_yaml
from skimage import io
from utils import psnr, psnr_img

from framework.common_callback import EpochCheckpoint, EpochTrainLog, EpochEvalLog
from accelerate.utils import set_seed
from framework.model import Model

# ############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")

parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--img_s', type=int, default=64, help="the slices of image sequence")
parser.add_argument('--img_w', type=int, default=256, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=256, help="the height of image sequence")
parser.add_argument('--gap_s', type=int, default=64, help='the slices of image gap')
parser.add_argument('--gap_w', type=int, default=128, help='the width of image gap')
parser.add_argument('--gap_h', type=int, default=128, help='the height of image gap')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam: weight_decay")

parser.add_argument('--normalize_bit', type=int, default=16)
parser.add_argument('--percentile', type=int, default=100)
parser.add_argument('--log', type=bool, default=False)

parser.add_argument('--log_folder', type=str, default='/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp15_level', help="output directory")
parser.add_argument('--datasets_path', type=str, default='/home/wwh/Dataset/simulation/', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='segmentation', help="A folder containing files for training")
parser.add_argument('--val_path', type=str, default='remove_bessel_bg_signal/test', help="dataset root path")
parser.add_argument('--remove_folder', type=list, default=[])
# parser.add_argument('--remove_files', type=list, default=['bg_removed_3.tif','bg_removed_4.tif','bg_removed_5.tif','bg_removed_6.tif',
#                                                           'raw_3.tif','raw_4.tif','raw_5.tif','raw_6.tif'])
parser.add_argument('--remove_files', type=list, default=[])
parser.add_argument('--select_img_num', type=int, default=1000, help='select the number of images')
parser.add_argument('--train_datasets_size', type=int, default=1000, help='datasets size for training')
parser.add_argument('--val_datasets_size', type=int, default=200, help='datasets size for training')
parser.add_argument('--train_stack_num', type=int, default=15, help='number of videos for training')

parser.add_argument('--checkpoint', type=str, default='/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp14_level/weights/0022_of_0030_checkpoint.pth')
opt = parser.parse_args()

print('the parameter of your training ----->')
print(opt)

''' <<<<<<<<<<<<<<<<<<<< setup >>>>>>>>>>>>>>>>> '''
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not os.path.exists(opt.log_folder):
    os.mkdir(opt.log_folder)

if not opt.datasets_folder:
    opt.datasets_folder = 'all'

yaml_name = opt.log_folder+'//para.yaml'
save_yaml(opt, yaml_name)

'''<<<<<<<<<<<<<<<<<<< dataset >>>>>>>>>>>>>>>>>>>>'''
transforms = Compose([
    RandomHorizontalFlipVideo(0.5)
])
paired_transform = PairedTransform(transforms)

train_set = crop_3D_dataset(opt, train=True, transform=paired_transform)
val_set = crop_3D_dataset(opt, train=False, transform=paired_transform)

# save dataset for faster loading at first time
with open('/home/wwh/code/DLT-main/dataset/train_set_level.pkl', 'wb') as f:
    pickle.dump(train_set, f)
with open('/home/wwh/code/DLT-main/dataset/val_set_level.pkl', 'wb') as f:
    pickle.dump(val_set, f)

# load dataset
# with open('/home/wwh/code/DLT-main/dataset/train_set_level.pkl', 'rb') as f:
#     train_set = pickle.load(f)
# with open('/home/wwh/code/DLT-main/dataset/val_set_level.pkl', 'rb') as f:
#     val_set = pickle.load(f)

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=1)

'''<<<<<<<<<<<<<<<<<<< loss function >>>>>>>>>>>>>>>>>>>>'''
loss_fn = MyLoss(mu=0.5, sigma=0.4, weight=False)

'''<<<<<<<<<<<<<<<<<<< network >>>>>>>>>>>>>>>>>>>>'''
network = Network_Uformer_3D_Unet(in_channels=3, out_channels=1, f_maps=16, layer_order='cgr', final_sigmoid=True)

''' <<<<<<<<<<< optimizer and lr_scheduler >>>>>>>>>>> '''
optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

total_steps = opt.n_epochs * len(train_loader)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=1 * opt.lr,
                                                   total_steps=total_steps, div_factor=25., pct_start=0.00)

''' <<<<<<<<<<< seed >>>>>>>>>>> '''
set_seed(42)

''' <<<<<<<<<<< metrics >>>>>>>>>>> '''
metrics = {"psnr": psnr, "psnr_mean_img": psnr_img}

''' <<<<<<<<<<< callback >>>>>>>>>>> '''
weights_folder = os.path.join(opt.log_folder, "weights")
os.makedirs(weights_folder, exist_ok=True)
val_loader_copy = deepcopy(val_loader)
callbacks = [
    EpochCheckpoint(weights_folder, save_freq=1),
    EpochTrainLog(opt.log_folder, save_freq=1),
    EpochEvalLog(opt.log_folder, save_freq=1, val_loader=val_loader_copy),
]

model = Model(network, loss_fn, optimizer, metrics_dict=metrics, lr_scheduler=lr_scheduler, opt=opt)
if opt.checkpoint:
    print('load checkpoint from %s' % opt.checkpoint)
    model.load_ckpt(opt.checkpoint)
history = model.fit(train_loader, val_loader, epochs=opt.n_epochs,
                        ckpt_path=os.path.join(weights_folder, "checkpoint.pth"),
                        callbacks=callbacks,
                        early_stopping=True,
                        patience=100,
                        monitor="val_psnr",
                        mode="max")





