import os
import torch
import pickle
import sys
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from copy import deepcopy

from torchvision.transforms import Compose

from loss import MyLoss
from dataset.bg_remove import crop_3D_dataset, RandomHorizontalFlipVideo, PairedTransform

import numpy as np
from utils import save_yaml
from skimage import io
from utils import psnr, psnr_img

from framework.common_callback import EpochCheckpoint, EpochTrainLog, EpochEvalLog
from accelerate.utils import set_seed
from framework.model import Model
import functools
from DWonder.RMBG.utils import FFDrealign4, inv_FFDrealign4
from DWonder.RMBG.Discriminator import NLayerDiscriminator3D
from DWonder.RMBG.network import Network_3D_Unet
from model_bank.network import Network_Uformer_3D_Unet


# ############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")

parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--img_w', type=int, default=176, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=176, help="the height of image sequence")
parser.add_argument('--img_s', type=int, default=176, help="the slices of image sequence")
parser.add_argument('--gap_s', type=int, default=200, help='the slices of image gap')
parser.add_argument('--gap_w', type=int, default=90, help='the width of image gap')
parser.add_argument('--gap_h', type=int, default=90, help='the height of image gap')

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam: weight_decay")
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")

parser.add_argument('--normalize_bit', type=int, default=12)
parser.add_argument('--percentile', type=float, default=99)

parser.add_argument('--log_folder', type=str, default='/home/wwh/code/DLT-main/log/bg_remove/unet_3d/exp10_RMBG_gan', help="output directory")
parser.add_argument('--datasets_path', type=str, default='/home/wwh/Dataset/simulation/', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='segmentation', help="A folder containing files for training")
parser.add_argument('--val_path', type=str, default='remove_bessel_bg_signal', help="dataset root path")
parser.add_argument('--remove_folder', type=list, default=[])
parser.add_argument('--remove_files', type=list, default=[])
parser.add_argument('--select_img_num', type=int, default=500, help='select the number of images')
parser.add_argument('--train_datasets_size', type=int, default=2000, help='datasets size for training')
parser.add_argument('--val_datasets_size', type=int, default=200, help='datasets size for training')

parser.add_argument('--checkpoint', type=str, default='')
opt = parser.parse_args()

print('the parameter of your training ----->')
print(opt)

''' <<<<<<<<<<<<<<<<<<<< setup >>>>>>>>>>>>>>>>> '''
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

if not os.path.exists(opt.log_folder):
    os.mkdir(opt.log_folder)
if not os.path.exists(opt.log_folder+'/weights'):
    os.makedirs(opt.log_folder+'/weights')
if not os.path.exists(opt.log_folder + '/m_outcome'):
    os.makedirs(opt.log_folder + '/m_outcome')

if not opt.datasets_folder:
    opt.datasets_folder = 'all'

yaml_name = opt.log_folder+'//para.yaml'
save_yaml(opt, yaml_name)
f_train = open(opt.log_folder+'//train_log.txt', 'w')
f_val = open(opt.log_folder+'//val_log.txt', 'w')

'''<<<<<<<<<<<<<<<<<<< dataset >>>>>>>>>>>>>>>>>>>>'''
transforms = Compose([
    RandomHorizontalFlipVideo(0.5)
])
paired_transform = PairedTransform(transforms)

# train_set = crop_3D_dataset(opt, train=True, transform=paired_transform)
# val_set = crop_3D_dataset(opt, train=False, transform=paired_transform)
#
# # save dataset for faster loading at first time
# with open('/home/wwh/code/DLT-main/dataset/train_set_preprocess_small.pkl', 'wb') as f:
#     pickle.dump(train_set, f)
# with open('/home/wwh/code/DLT-main/dataset/val_set_preprocess_small.pkl', 'wb') as f:
#     pickle.dump(val_set, f)

# load dataset
with open('/home/wwh/code/DLT-main/dataset/train_set_preprocess_small.pkl', 'rb') as f:
    train_set = pickle.load(f)
with open('/home/wwh/code/DLT-main/dataset/val_set_preprocess_small.pkl', 'rb') as f:
    val_set = pickle.load(f)

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=2)

'''<<<<<<<<<<<<<<<<<<< loss function >>>>>>>>>>>>>>>>>>>>'''
L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()
Dloss = torch.nn.BCEWithLogitsLoss()

'''<<<<<<<<<<<<<<<<<<< network >>>>>>>>>>>>>>>>>>>>'''
denoise_generator = Network_Uformer_3D_Unet(in_channels=4, out_channels=4, f_maps=32, layer_order='cgr', final_sigmoid=True)
norm_layer = functools.partial(torch.nn.BatchNorm3d, affine=True, track_running_stats=True)
Dnet = NLayerDiscriminator3D(input_nc=4, ndf=16, n_layers=3, norm_layer=norm_layer)
if torch.cuda.is_available():
    denoise_generator = denoise_generator.cuda()
    Dnet.cuda()
    print('\033[1;31mUsing {} GPU for training -----> \033[0m'.format(torch.cuda.device_count()))
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()
    Dloss.cuda()
''' <<<<<<<<<<< optimizer and lr_scheduler >>>>>>>>>>> '''
optimizer_G = torch.optim.Adam(denoise_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

for epoch in range(0, opt.n_epochs):
    n = train_loader.size if hasattr(train_loader, 'size') else len(train_loader)
    loop = tqdm(enumerate(train_loader, start=1), total=n, file=sys.stdout, ncols=140)
    for step, batch in loop:
        input, target = batch
        denoise_generator.train()
        real_A = FFDrealign4(input).cuda()
        real_B = FFDrealign4(target).cuda()
        real_A = Variable(real_A)

        fake_B = denoise_generator(real_A)

        # upgrade Dnet
        D_fake_B = Dnet(fake_B.detach())
        D_real_B = Dnet(real_B)
        valid = Variable(torch.cuda.FloatTensor(D_fake_B.shape).fill_(1.0), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(D_fake_B.shape).fill_(0.0), requires_grad=False)

        optimizer_D.zero_grad()
        real_loss = Dloss(D_real_B, valid)
        fake_loss = Dloss(D_fake_B, fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # upgrade Gnet
        D_fake_B = Dnet(fake_B)
        Dloss1 = Dloss(D_fake_B, valid)
        L1_loss = L1_pixelwise(fake_B, real_B)
        L2_loss = L2_pixelwise(fake_B, real_B)
        optimizer_G.zero_grad()
        Total_loss = L1_loss + 0.5 * L2_loss + epoch / opt.n_epochs * Dloss1
        Total_loss.backward()
        optimizer_G.step()

        if step % 10 == 0:
            fake_B_realign = inv_FFDrealign4(fake_B)
            real_B_realign = inv_FFDrealign4(real_B)
            psnr_val = psnr(fake_B_realign, real_B_realign, format='pre', bit=opt.normalize_bit)
            print(
                '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f] [PSNR: %.2f]'
                % (
                    epoch + 1, opt.n_epochs,
                    step + 1, n,
                    Total_loss.item(),
                    L1_loss.item(),
                    L2_loss.item(),
                    psnr_val
                ), end=' ')
            f_train.write('Total_loss ' + str(Total_loss.item()) + ' psnr_val ' + str(psnr_val) + ' \n')
            print('Dloss1 ' + str(Dloss1.item()) + ' fake_loss ' + str(fake_loss.item()) + ' \n')

    weights_name = "checkpoint.pth"
    ckpt_path = os.path.join(opt.log_folder+'/weights', f'{epoch:0>4d}_of_{opt.n_epochs:0>4d}_{weights_name}')
    torch.save(denoise_generator.state_dict(), ckpt_path)

    loop_val = tqdm(enumerate(val_loader, start=1), total=len(val_loader), file=sys.stdout, ncols=140)
    psnr_list = []
    for step, batch in loop_val:
        input, target = batch
        real_A = FFDrealign4(input).cuda()
        real_B = FFDrealign4(target).cuda()
        denoise_generator.eval()
        with torch.no_grad():
            fake_B = denoise_generator(real_A)
        fake_B_realign = inv_FFDrealign4(fake_B)
        real_B_realign = inv_FFDrealign4(real_B)
        psnr_val = psnr(fake_B_realign, real_B_realign, format='pre', bit=opt.normalize_bit)
        psnr_list.append(psnr_val)
        print('\r[Epoch %d/%d]  [PSNR: %.2f]' % (epoch + 1, opt.n_epochs, psnr_val), end=' ')
    f_val.write('psnr_val ' + str(np.mean(psnr_list)) + ' \n')

    output_img = fake_B_realign.cpu().detach().numpy()
    train_GT = real_B_realign.cpu().detach().numpy()
    real_A_realign = inv_FFDrealign4(real_A)
    train_input = real_A_realign.cpu().detach().numpy()

    output_img = np.clip(np.exp2(output_img.squeeze() * opt.normalize_bit), 0, 2 ** opt.normalize_bit).astype('uint16')
    gt_img = np.clip(np.exp2(train_GT.squeeze() * opt.normalize_bit), 0, 2 ** opt.normalize_bit).astype('uint16')
    output_name = opt.log_folder + '/m_outcome' + '/' + str(epoch) + '_output.tif'
    gt_name = opt.log_folder + '/m_outcome' + '/' + str(epoch) + '_gt.tif'
    io.imsave(output_name, output_img)
    io.imsave(gt_name, gt_img)

f_train.close()
f_val.close()










