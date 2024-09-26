import numpy
import numpy as np
import os
import tifffile as tiff

import math
from utils import normalize_0_1, normalize_log_0_1, normalize_log_std, preprocess, process_video_block
from glob import glob
from torch.utils.data import Dataset
import h5py
import torch
import random
import pickle
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


def get_gap_s(args, img, stack_num, data_size):
    whole_w = img.shape[2]
    whole_h = img.shape[1]
    whole_s = img.shape[0]
    print('whole_w -----> ', whole_w)
    print('whole_h -----> ', whole_h)
    print('whole_s -----> ', whole_s)
    w_num = math.floor((whole_w-args.img_w)/args.gap_w)+1
    h_num = math.floor((whole_h-args.img_h)/args.gap_h)+1
    s_num = math.ceil(data_size/w_num/h_num/stack_num)
    print('w_num -----> ', w_num)
    print('h_num -----> ', h_num)
    print('s_num -----> ', s_num)
    gap_s = math.floor((whole_s-args.img_s)/(s_num-1))
    print('gap_s -----> ', gap_s)
    return gap_s


def get_stack_num(folder_paths: list):
    stack_num = 0
    for folder_path in folder_paths:
        img_paths = glob(folder_path+'/*.tif')
        stack_num += len(img_paths)
    return stack_num / 2


def preprocess_lessMemoryMulStacks(args, train: bool):
    img_h = args.img_h
    img_w = args.img_w
    img_s = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w

    # 适用于将不同条件的数据放在不同文件夹的情况，比如不同的bessel beam
    if train:
        if args.datasets_folder == 'all':
            im_folder = list(os.listdir(args.datasets_path))
            im_folders = [args.datasets_path + i for i in im_folder if i not in args.remove_folder]
        else:
            im_folders = [args.datasets_path+'/'+args.datasets_folder]
    else:
        im_folders = [args.datasets_path+'/'+args.val_path]

    name_list = []
    coordinate_list = {}
    clean_movs = {}
    raw_movs = {}

    stack_num = min(get_stack_num(im_folders), args.train_stack_num)
    print('stack_num -----> ', stack_num)
    for im_folder in im_folders:
        for im_dir in glob(im_folder + '/*.tif'):
            im_name = os.path.basename(im_dir)
            # 只用了命名上小于train_stack_num的视频
            if im_name in args.remove_files or int(im_name.split('.')[0].split('_')[-1]) > args.train_stack_num:
                continue
            label = os.path.basename(im_folder) + '_' + im_name.replace('.tif', '')
            movie = tiff.imread(im_dir)
            if movie.shape[0] > args.select_img_num:
                movie = movie[-args.select_img_num-1:-1, :, :]
            if train:
                if 'clean' in im_name:
                    # movie = normalize_0_1(movie)
                    # movie = normalize_0_1(movie, percentile=args.percentile)
                    # movie = normalize_log_std(movie, gt=True)
                    # movie = normalize_log_0_1(movie, gt=True)
                    # movie = preprocess(movie, gt=True, percentile=args.percentile, bit=args.normalize_bit, log=args.log)
                    movie_ = preprocess(movie, gt=True, percentile=args.percentile, bit=args.normalize_bit, log=args.log)
                    clean_movs[label] = movie_.astype(np.float32)
                    continue
            else:
                if 'bg_removed' in im_name:
                    # movie = normalize_0_1(movie)
                    # movie = normalize_0_1(movie, percentile=args.percentile)
                    # movie = normalize_log_std(movie, gt=True)
                    # movie = normalize_log_0_1(movie, gt=True)
                    # movie = preprocess(movie, gt=True, percentile=args.percentile, bit=args.normalize_bit, log=args.log)
                    movie_ = preprocess(movie, gt=True, percentile=args.percentile, bit=args.normalize_bit, log=args.log)
                    clean_movs[label] = movie_.astype(np.float32)
                    continue
            if 'raw' in im_name:
                # movie = normalize_0_1(movie)
                # movie = normalize_log_std(movie, gt=False)
                # movie = normalize_log_0_1(movie, gt=False)
                # movie = preprocess(movie, gt=False, percentile=args.percentile, bit=args.normalize_bit, log=args.log)
                movie_ = process_video_block(movie)
                raw_movs[label] = movie_.astype(np.float32)

            whole_w = movie.shape[2]
            whole_h = movie.shape[1]
            whole_s = movie.shape[0]

            if train:
                gap_s = get_gap_s(args, movie, stack_num, args.train_datasets_size)
            else:
                gap_s = get_gap_s(args, movie, stack_num, args.val_datasets_size)

            for x in range(0, int((whole_h-img_h+gap_h)/gap_h)):
                for y in range(0, int((whole_w-img_w+gap_w)/gap_w)):
                    for z in range(0, int((whole_s-img_s+gap_s)/gap_s)):
                        single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                        init_h = gap_h*x
                        end_h = gap_h*x + img_h
                        init_w = gap_w*y
                        end_w = gap_w*y + img_w
                        init_s = gap_s*z
                        end_s = gap_s*z + img_s
                        single_coordinate['init_h'] = init_h
                        single_coordinate['end_h'] = end_h
                        single_coordinate['init_w'] = init_w
                        single_coordinate['end_w'] = end_w
                        single_coordinate['init_s'] = init_s
                        single_coordinate['end_s'] = end_s
                        # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                        patch_name = label+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                        # train_raw.append(noise_patch1.transpose(1,2,0))
                        name_list.append(patch_name)
                        # print(' single_coordinate -----> ',single_coordinate)
                        coordinate_list[patch_name] = single_coordinate

    return name_list, coordinate_list, raw_movs, clean_movs


def test_preprocess_lessMemoryNoTail(args, format):
    img_h = args.img_h
    img_w = args.img_w
    img_s = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s = args.gap_s
    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    noise_im = np.array(tiff.imread(args.datasets_path))
    index = os.path.splitext(os.path.basename(args.datasets_path))[0].split('_')[-1]
    gt_path = '/home/wwh/Dataset/simulation/segmentation/clean_20um_' + index + '.tif'
    gt_im = np.array(tiff.imread(gt_path))
    # with open('/home/wwh/code/DLT-main/dataset/raw_' + index + '.pkl', 'rb') as f:
    #     noise_im = pickle.load(f)
    # with open('/home/wwh/code/DLT-main/dataset/bg_removed_' + index + '.pkl', 'rb') as f:
    #     gt_im = pickle.load(f)

    name_list = []
    coordinate_list = {}

    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[-args.test_datasize-1: -1, :, :]

    # if format == 'std':
    #     noise_im = normalize_log_std(noise_im, gt=False)
    # elif format == 'max':
    #     noise_im = normalize_log_0_1(noise_im, gt=False)
    # elif format == 'pre':
    #     noise_im = preprocess(noise_im, gt=False, percentile=args.percentile, bit=args.normalize_bit)
    # noise_im = process_video_block(noise_im)

    if gt_im.shape[0] > args.test_datasize:
        gt_im = gt_im[-args.test_datasize-1: -1, :, :]

    # if format == 'std':
    #     gt_im = normalize_log_std(gt_im, gt=True)
    # elif format == 'max':
    #     gt_im = normalize_log_0_1(gt_im, gt=True)
    # elif format == 'pre':
    #     gt_im = preprocess(gt_im, gt=True, percentile=args.percentile, bit=args.normalize_bit)
    # gt_im = preprocess(gt_im, gt=True, percentile=args.percentile, bit=args.normalize_bit, log=args.log)

    whole_w = noise_im.shape[3]
    whole_h = noise_im.shape[2]
    whole_s = noise_im.shape[1]

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
    return name_list, noise_im, gt_im, coordinate_list


class RandomCropVideo:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, video):
        c, t, h, w = video.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        return video[:, :, top: top + self.crop_size, left: left + self.crop_size]


class RandomHorizontalFlipVideo:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        if random.random() < self.p:
            return torch.flip(video, [-1])  # 沿宽度维度翻转
        return video


class PairedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        img1_transformed = self.transform(img1)

        random.seed(seed)
        img2_transformed = self.transform(img2)

        return img1_transformed, img2_transformed


class crop_3D_dataset(Dataset):
    def __init__(self, arg, train: bool, transform=None):
        self.name_list, self.coordinate_list, self.raw_movs, self.clean_movs = \
            preprocess_lessMemoryMulStacks(arg, train)
        self.transform = transform
        self.train = train
        self.arg = arg

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        single_coordinate = self.coordinate_list[name]
        temp = name.split('_')
        if self.train:
            raw_label = '_'.join(temp[:-3])
            temp[-6] = 'clean'
            clean_label = '_'.join(temp[:-3])
        else:
            raw_label = '_'.join(temp[:-3])
            temp[-5] = 'bg_removed'
            clean_label = '_'.join(temp[:-3])
        noise_img = self.raw_movs[raw_label]
        clean_img = self.clean_movs[clean_label]

        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        # noise_patch = noise_img[:, init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        clean_patch = clean_img[init_s:end_s, init_h:end_h, init_w:end_w]

        noise_patch = process_video_block(noise_patch)
        # noise_patch = preprocess(noise_patch, gt=False, percentile=self.arg.percentile,
        #                          bit=self.arg.normalize_bit, log=self.arg.log)
        clean_patch = preprocess(clean_patch, gt=True, percentile=self.arg.percentile,
                                 bit=self.arg.normalize_bit, log=self.arg.log)

        # noise_patch = torch.from_numpy(np.expand_dims(noise_patch.astype(np.float32), 0))
        noise_patch = torch.from_numpy(noise_patch.astype(np.float32))
        clean_patch = torch.from_numpy(np.expand_dims(clean_patch.astype(np.float32), 0))

        if self.transform:
            noise_patch, clean_patch = self.transform(noise_patch, clean_patch)

        return noise_patch, clean_patch