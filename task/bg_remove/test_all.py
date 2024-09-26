import os
import torch
import torch.nn as nn
import argparse
import time
import datetime
from dataset.bg_remove import test_preprocess_lessMemoryNoTail
from model_bank.network import Network_3D_Unet, Network_3D_Nnet, Network_Uformer_3D_Nnet, Network_Uformer_3D_Unet
from utils import psnr, psnr_img

import numpy as np
from skimage import io
import pickle
from utils import preprocess, process_video_block, find_all_test_files

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

parser.add_argument('--output_dir', type=str,
                    default='/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp15_level/test_results/')
parser.add_argument('--datasets_path', type=str,
                    default='/home/wwh/Dataset/simulation/segmentation/raw_20um_108.tif')
parser.add_argument('--pth_path', type=str,
                    default='/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp17_level/weights/0040_of_0040_checkpoint.pth')
parser.add_argument('--test_datasize', type=int, default=2000, help='dataset size to be tested')

opt = parser.parse_args()
########################################################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_format = 'no_log'

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

files = find_all_test_files(train_stack_num=25)

# network = Network_3D_Unet(in_channels=1, out_channels=1, layer_order='cgr', f_maps=16, final_sigmoid=True)
network = Network_Uformer_3D_Unet(in_channels=3, out_channels=1, f_maps=16, layer_order='cgr', final_sigmoid=True)
network.load_state_dict(torch.load(opt.pth_path, map_location="cpu"))
network.cuda()

for path in files:
    opt.datasets_path = path

    name_list, noise_img, _, coordinate_list = test_preprocess_lessMemoryNoTail(opt, data_format)

    denoise_img = np.zeros(noise_img.shape[1:])

    for index in range(len(name_list)):

        single_coordinate = coordinate_list[name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = noise_img[:, init_s:end_s, init_h:end_h, init_w:end_w]
        # normalization
        # noise_patch = process_video_block(noise_patch_)

        noise_patch = torch.from_numpy(np.expand_dims(noise_patch.astype(np.float32), 0)).cuda()

        input_name = name_list[index]
        print(' input_name -----> ', input_name)
        print(' single_coordinate -----> ', single_coordinate)
        print(' noise_patch -----> ', noise_patch.shape)

        network.eval()
        with torch.no_grad():
            bg_removed_patch = network(noise_patch)[0]

        ############################################################################################################

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

    normalize_factor = 2 ** opt.normalize_bit - 1
    if data_format == 'no_log':
        output_img = np.clip(denoise_img.squeeze() * normalize_factor, 0, normalize_factor).astype('uint16')

    result_name = opt.output_dir + '//' + 'raw_' + path.split('_')[-1].split('.')[0] + '.tif'
    io.imsave(result_name, output_img)


