import os
import torch
import torch.nn as nn
import argparse
import time
import datetime
from model_bank.network import Network_Uformer_3D_Unet
from utils import psnr, psnr_img

import numpy as np
from utils import save_yaml
from skimage import io
import tifffile as tiff
import math
import pickle
from utils import preprocess, process_video_block

from framework.model import Model
######################################################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--img_s', type=int, default=64, help="the slices of image sequence")
parser.add_argument('--img_w', type=int, default=256, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=256, help="the height of image sequence")
parser.add_argument('--gap_s', type=int, default=64, help='the slices of image gap')
parser.add_argument('--gap_w', type=int, default=128, help='the width of image gap')
parser.add_argument('--gap_h', type=int, default=128, help='the height of image gap')

parser.add_argument('--normalize_bit', type=int, default=16)
parser.add_argument('--percentile', type=int, default=100)
parser.add_argument('--log', type=bool, default=False)

parser.add_argument('--output_dir', type=str, default='/home/disks/sde/wwh/wuwh/code/DLT-main/task/test_fish/log')
parser.add_argument('--datasets_path', type=str, default='/home/disks/sde/wwh/wuwh/Dataset/Fish_for_Weihang/5_min_bessel_low_power/Bliq_VMS/C2/combined_video.tif')
parser.add_argument('--pth_path', type=str, default='/home/disks/sde/wwh/wuwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp17_local_normalization_level/weights/0030_of_0030_checkpoint.pth')
parser.add_argument('--test_datasize', type=int, default=1000, help='dataset size to be tested')

opt = parser.parse_args()
debug = False
########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def dataset(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s = args.gap_s
    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    noise_im = np.array(tiff.imread(args.datasets_path)).squeeze()

    name_list = []
    coordinate_list = {}

    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[-args.test_datasize-1: -1, :, :]

    # noise_im = process_video_block(noise_im)

    whole_w = noise_im.shape[2]
    whole_h = noise_im.shape[1]
    whole_s = noise_im.shape[0]

    num_w = math.ceil((whole_w-img_w)/gap_w + 1)
    num_h = math.ceil((whole_h-img_h)/gap_h + 1)
    num_s = math.ceil((whole_s-img_s)/gap_s + 1)

    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s*z
                    end_s = gap_s*z + img_s
                elif z == (num_s-1):
                    init_s = whole_s - img_s
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y*gap_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = img_w-cut_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x*gap_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = img_h-cut_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z*gap_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = img_s-cut_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s
                else:
                    single_coordinate['stack_start_s'] = z*gap_s+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s-cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = os.path.basename(args.datasets_path)+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate
    return name_list, noise_im, coordinate_list


name_list, noise_img, coordinate_list = dataset(opt)


if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

network = Network_Uformer_3D_Unet(in_channels=3, out_channels=1, f_maps=16, layer_order='cgr', final_sigmoid=True)
network.load_state_dict(torch.load(opt.pth_path, map_location="cpu"))
network.cuda()

denoise_img = np.zeros(noise_img.shape)
input_img = np.zeros(noise_img.shape)

for index in range(len(name_list)):

    single_coordinate = coordinate_list[name_list[index]]
    init_h = single_coordinate['init_h']
    end_h = single_coordinate['end_h']
    init_w = single_coordinate['init_w']
    end_w = single_coordinate['end_w']
    init_s = single_coordinate['init_s']
    end_s = single_coordinate['end_s']
    noise_patch_ = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
    # normalization
    noise_patch = process_video_block(noise_patch_)

    noise_patch = torch.from_numpy(np.expand_dims(noise_patch.astype(np.float32), 0)).cuda()

    input_name = name_list[index]
    print(' input_name -----> ', input_name)
    print(' single_coordinate -----> ', single_coordinate)
    print(' noise_patch -----> ', noise_patch.shape)

    network.eval()
    with torch.no_grad():
        bg_removed_patch = network(noise_patch)[0]

    ###########################################################################################################

    output_image = np.squeeze(bg_removed_patch.cpu().detach().numpy())

    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
        = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
        = noise_patch_[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]


normalize_factor = 2 ** opt.normalize_bit - 1

output_img = np.clip(denoise_img.squeeze() * normalize_factor, 0, normalize_factor).astype('uint16')


result_name = opt.output_dir + '//' + 'output.tif'
input_name = opt.output_dir + '//' + 'input.tif'
io.imsave(result_name, output_img)
io.imsave(input_name, input_img)


