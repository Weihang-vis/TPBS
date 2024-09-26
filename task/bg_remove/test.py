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
from utils import save_yaml
from skimage import io
import pickle
# from DWonder.RMBG.utils import FFDrealign4, inv_FFDrealign4
# from DWonder.RMBG.Discriminator import NLayerDiscriminator3D
# from DWonder.RMBG.network import Network_3D_Unet
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

parser.add_argument('--output_dir', type=str, default='/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp12_local_normalization_level/test_results/raw_108')
parser.add_argument('--datasets_path', type=str, default='/home/wwh/Dataset/simulation/segmentation/raw_20um_108.tif')
parser.add_argument('--pth_path', type=str, default='/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp12_local_normalization_level/weights/0026_of_0030_checkpoint.pth')
parser.add_argument('--test_datasize', type=int, default=2000, help='dataset size to be tested')

opt = parser.parse_args()
debug = False
########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_format = 'no_log'

# name_list, noise_img, gt_img, coordinate_list = test_preprocess_lessMemoryNoTail(opt, data_format)
# data_to_save = {'name_list': name_list, 'coordinate_list': coordinate_list, 'noise_img': noise_img, 'gt_img': gt_img}
# with open('/home/wwh/code/DLT-main/dataset/test_108.pkl', 'wb') as f:
#     pickle.dump(data_to_save, f)

# load data
with open('/home/wwh/code/DLT-main/dataset/test_108.pkl', 'rb') as f:
    data = pickle.load(f)
    name_list, noise_img, gt_img, coordinate_list = data['name_list'], data['noise_img'], data['gt_img'], \
                                                       data['coordinate_list']

if not os.path.exists(opt.output_dir): 
    os.makedirs(opt.output_dir)

# network = Network_Uformer_3D_Nnet(in_channels=1, out_channels=1, f_maps=64, final_sigmoid=True)
# network = Network_3D_Unet(in_channels=1, out_channels=1, layer_order='cgr', f_maps=16, final_sigmoid=True)
# network = Network_3D_Unet(in_channels=4, out_channels=4, f_maps=16, final_sigmoid=True)
network = Network_Uformer_3D_Unet(in_channels=3, out_channels=1, f_maps=16, layer_order='cgr', final_sigmoid=True)
network.load_state_dict(torch.load(opt.pth_path, map_location="cpu"))
network.cuda()

denoise_img = np.zeros(noise_img.shape)
input_img = np.zeros(noise_img.shape)
gt_img = np.zeros(noise_img.shape)
psnr_list = []
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

    gt_patch = gt_img[init_s:end_s, init_h:end_h, init_w:end_w]
    gt_patch = preprocess(gt_patch, gt=True, percentile=opt.percentile, bit=opt.normalize_bit, log=opt.log)

    input_name = name_list[index]
    print(' input_name -----> ', input_name)
    print(' single_coordinate -----> ', single_coordinate)
    print(' noise_patch -----> ', noise_patch.shape)

    network.eval()
    with torch.no_grad():
        bg_removed_patch = network(noise_patch)[0]
        # noise_patch = FFDrealign4(noise_patch).cuda()
        # bg_removed_patch = network(noise_patch)
        # bg_removed_patch = inv_FFDrealign4(bg_removed_patch)
        # noise_patch = inv_FFDrealign4(noise_patch)

    ############################################################################################################
    if debug is True:
        gt_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(gt_patch, 3), 0)).cuda()
        psnr_list.append(psnr(bg_removed_patch, gt_tensor))

    output_image = np.squeeze(bg_removed_patch.cpu().detach().numpy())
    # raw_image = np.squeeze(noise_patch.cpu().detach().numpy())

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
    gt_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
        = gt_patch[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

# for 0-1 normalization
# output_img = np.clip(denoise_img.squeeze().astype(np.float32)*opt.normalize_factor, 0, 65535).astype('uint16')
# input_img = np.clip(input_img.squeeze().astype(np.float32)*opt.normalize_factor, 0, 65535).astype('uint16')

# for log 0-1_max normalization
# output_img = np.clip(np.exp(denoise_img.squeeze() * 6.30266078), 0, 65535).astype('uint16')
# input_img = np.clip(np.exp(input_img.squeeze() * 10.979528), 0, 65535).astype('uint16')

# for log std normalization
# gt_mean, gt_std, gt_max = 1.44140275, 0.47881246, 6.6512957
gt_mean, gt_std, gt_max = 1.5690196, 0.6133263, 7.3532133

normalize_factor = 2 ** opt.normalize_bit - 1
if data_format == 'std':
    output_img = np.clip(np.exp(denoise_img.squeeze() * gt_std + gt_mean), 0, normalize_factor).astype('uint16')
    # input_img = np.clip(np.exp(input_img.squeeze() * pred_std + pred_mean), 0, 65535).astype('uint16')
    gt_img = np.clip(np.exp(gt_img.squeeze() * gt_std + gt_mean), 0, normalize_factor).astype('uint16')
elif data_format == 'max':
    output_img = np.clip(np.exp(denoise_img.squeeze() * gt_max), 0, normalize_factor).astype('uint16')
    gt_img = np.clip(np.exp(gt_img.squeeze() * gt_max), 0, normalize_factor).astype('uint16')
elif data_format == 'pre':
    output_img = np.clip(np.exp2(denoise_img.squeeze() * opt.normalize_bit), 0, normalize_factor).astype('uint16')
    gt_img = np.clip(np.exp2(gt_img.squeeze() * opt.normalize_bit), 0, normalize_factor).astype('uint16')
elif data_format == 'no_log':
    output_img = np.clip(denoise_img.squeeze() * normalize_factor, 0, normalize_factor).astype('uint16')
    gt_img = np.clip(gt_img.squeeze() * normalize_factor, 0, normalize_factor).astype('uint16')

# psnr_val = psnr(output_img, gt_img, format='no_log')
# print('psnr_val -----> ', psnr_val)


result_name = opt.output_dir + '//' + 'output.tif'
# input_name = opt.output_dir + '//' + 'input.tif'
gt_name = opt.output_dir + '//' + 'gt.tif'
io.imsave(result_name, output_img)
# io.imsave(input_name, input_img)
io.imsave(gt_name, gt_img)

