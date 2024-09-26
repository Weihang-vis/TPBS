import numpy as np
import torch
import yaml
import os
import re


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


def normalize_0_1(x: np.ndarray, percentile=100):
    if percentile != 100:
        max_value = np.percentile(x, percentile)
        return (x - x.min()).astype(np.float32) / (max_value - x.min())
    else:
        return (x - x.min()).astype(np.float32) / (x.max() - x.min())


def normalize_log_0_1(x: np.ndarray, gt: bool):
    x[x < 0] = 0
    x = np.log(x + 1)
    if gt:
        maximum = 7.3532133  # 6.6512957
    else:
        maximum = 11.954944  # 11.294023
    # return (x - x.min()).astype(np.float32) / (x.max() - x.min())
    return x / maximum


def normalize_log_std(x: np.ndarray, gt: bool):
    x[x < 0] = 0
    x = np.log(x + 1)
    if gt:
        mean = 1.5690196  # 1.44140275
        std = 0.6133263  # 0.47881246
    else:
        mean = 7.896325  # 7.64619605
        std = 0.18054935  # 0.19071723
    return (x - mean) / std


def preprocess(x: np.ndarray, gt: bool, percentile=100, bit=16, log=False):
    if not gt:
        max_value = np.percentile(x, percentile)
        x[x < 0] = 0
        x[x > max_value] = max_value
        # requantify to 16 bits
        x = (x / max_value * (2 ** bit - 1)).astype(np.uint32)
        # logarithmic transformation
        if log:
            x = np.log2(x + 1)
            return (x / bit).astype(np.float32)
        else:
            x = x / (2 ** bit - 1)
            return x.astype(np.float32)
    else:
        x[x < 0] = 0
        x = (x / x.max() * (2 ** bit - 1)).astype(np.uint32)
        if log:
            x = np.log2(x + 1)
            return (x / bit).astype(np.float32)
        else:
            x = x / (2 ** bit - 1)
            return x.astype(np.float32)


def process_video_block(video_block, gt=False, raw_bit=16, gt_bit=10):
    video_block = (video_block / video_block.max() * (2 ** raw_bit - 1)).astype(np.uint32)
    # 定义像素值的区间
    # ranges = [(0, 254), (255, 4095), (4096, 35535)]
    if not gt:
        ranges = [(0, 4095), (4096, 16383), (16384, 2**raw_bit-1)]
    else:
        ranges = [(0, 255), (256, 511), (512, 2**gt_bit-1)]

    # 初始化一个新的视频块，三个通道
    processed_block = np.zeros((3, video_block.shape[0], video_block.shape[1], video_block.shape[2]))

    # 对每个区间处理
    for channel, (low, high) in enumerate(ranges):
        # 截断像素值到指定区间
        truncated = np.clip(video_block, low, high)

        # 如果不是第一个区间，则需要减去下界
        if low != 0:
            truncated -= low

        # 归一化到0-1范围
        normalized = truncated / (high - low)

        # 填充到新的视频块
        processed_block[channel] = normalized

    return processed_block


def save_yaml(opt, yaml_name):
    para = {'n_epochs':0, 'datasets_folder':0, 'log_folder':0, 'batch_size':0,
            'img_s':0,'img_w':0, 'img_h':0, 'gap_h':0,'gap_w':0,'gap_s':0,'lr':0,'normalize_factor':0}

    para["n_epochs"] = opt.n_epochs
    para["datasets_folder"] = opt.datasets_folder
    para["output_dir"] = opt.log_folder
    para["batch_size"] = opt.batch_size
    para["img_s"] = opt.img_s
    para["img_w"] = opt.img_w
    para["img_h"] = opt.img_h
    para["gap_h"] = opt.gap_h
    para["gap_w"] = opt.gap_w
    para["gap_s"] = opt.gap_s
    para["lr"] = opt.lr
    para["normalize_bit"] = opt.normalize_bit
    para["datasets_path"] = opt.datasets_path
    para["train_datasets_size"] = opt.train_datasets_size
    with open(yaml_name, 'w') as f:
        data = yaml.dump(para, f)


def psnr(im1, im2, format='no_log', bit=16, **kwargs):
    normalize_factor = 2 ** bit - 1

    # gt_mean, gt_std, gt_max = 1.44140275, 0.47881246, 6.6512957
    gt_mean, gt_std, gt_max = 1.5690196, 0.6133263, 7.3532133

    if isinstance(im1[0], torch.Tensor):

        im1 = im1[0].cpu().detach().numpy().squeeze().astype(np.float32)
        im2 = im2.cpu().detach().numpy().squeeze().astype(np.float32)
        if format == 'std':
            im1 = np.clip(np.exp(im1 * gt_std + gt_mean), 0, normalize_factor).astype('uint16')
            im2 = np.clip(np.exp(im2 * gt_std + gt_mean), 0, normalize_factor).astype('uint16')
        elif format == 'max':
            im1 = np.clip(np.exp(im1 * gt_max), 0, normalize_factor).astype('uint16')
            im2 = np.clip(np.exp(im2 * gt_max), 0, normalize_factor).astype('uint16')
        elif format == 'pre':
            im1 = np.clip(np.exp2(im1 * bit), 0, normalize_factor).astype('uint16')
            im2 = np.clip(np.exp2(im2 * bit), 0, normalize_factor).astype('uint16')
        elif format == 'no_log':
            im1 = np.clip(im1 * normalize_factor, 0, normalize_factor).astype('uint16')
            im2 = np.clip(im2 * normalize_factor, 0, normalize_factor).astype('uint16')

    mse = ((im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(normalize_factor ** 2 / mse)
    return psnr


def psnr_img(im1, im2, format='no_log', bit=16, **kwargs):
    normalize_factor = 2 ** bit - 1

    # gt_mean, gt_std, gt_max = 1.44140275, 0.47881246, 6.6512957
    gt_mean, gt_std, gt_max = 1.5690196, 0.6133263, 7.3532133

    if isinstance(im1[1], torch.Tensor):
        im1 = im1[1].cpu().detach().numpy().squeeze().astype(np.float32)
        im2 = im2.mean(2, keepdim=True).cpu().detach().numpy().squeeze().astype(np.float32)

        if format == 'std':
            im1 = np.clip(np.exp(im1 * gt_std + gt_mean), 0, normalize_factor).astype('uint16')
            im2 = np.clip(np.exp(im2 * gt_std + gt_mean), 0, normalize_factor).astype('uint16')
        elif format == 'max':
            im1 = np.clip(np.exp(im1 * gt_max), 0, normalize_factor).astype('uint16')
            im2 = np.clip(np.exp(im2 * gt_max), 0, normalize_factor).astype('uint16')
        elif format == 'pre':
            im1 = np.clip(np.exp2(im1 * bit), 0, normalize_factor).astype('uint16')
            im2 = np.clip(np.exp2(im2 * bit), 0, normalize_factor).astype('uint16')
        elif format == 'no_log':
            im1 = np.clip(im1 * normalize_factor, 0, normalize_factor).astype('uint16')
            im2 = np.clip(im2 * normalize_factor, 0, normalize_factor).astype('uint16')

    mse = ((im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(normalize_factor ** 2 / mse)
    return psnr


def corr(im1, im2, **kwargs):
    pass


def find_all_test_files(train_stack_num=25):

    folder_path = '/home/wwh/Dataset/simulation/segmentation'
    file_list = []

    # 遍历文件夹
    for filename in os.listdir(folder_path):
        # 检查文件名是否符合条件
        if filename.startswith('raw') and filename.endswith('.tif'):
            # 将文件名添加到列表中
            file_list.append(filename)

    file_list.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))

    # 去除前train_stack_num个文件
    file_list = file_list[train_stack_num:]

    # 去除被用作yolo训练集的视频
    yolo_train_folder = '/home/wwh/code/yolov8/datasets/bessel20um/raw/labels/train'
    yolo_train_list = sorted(os.listdir(yolo_train_folder), key=lambda x: int(x.split('.')[0]))

    def remove_leading_zeros(filename):
        # Extract the number from the filename
        number = re.search(r'\d+', filename)
        return int(number.group()) if number else None

    idx = [remove_leading_zeros(fn) + 1 for fn in yolo_train_list]
    file_list = [i for i in file_list if int(i.split('_')[-1].split('.')[0]) not in idx]
    file_path_list = [os.path.join(folder_path, i) for i in file_list]

    return file_path_list