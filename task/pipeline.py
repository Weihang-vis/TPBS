"""
Some function used in pipeline
"""
import h5py
import numpy as np
import os


def yolo2json(path):
    pass


def is_similar(line1, line2, tolerance=0.00001):
    # 分割行以获取各个元素
    elements1 = line1.strip().split()
    elements2 = line2.strip().split()

    # 比较每个坐标值，考虑容忍度
    for a, b in zip(elements1[1:], elements2[1:]):
        if abs(float(a) - float(b)) > tolerance:
            return False

    return True


def find_clean_gt(train:bool, ind=107, tolerance=0.00001):
    """
    Find the clean gt index in the gt list
    to match each clean gt mask with the corresponding anchor
    :param ind: the index of the video
    :return: the list of clean gt index in the gt list
    """
    if train:
        gt_path = '/home/wwh/Dataset/simulation/segmentation/labels/train'
        gt_clean_path = '/home/wwh/Dataset/simulation/segmentation/labels/train/modify'
    else:
        gt_path = '/home/wwh/Dataset/simulation/segmentation/labels/val'
        gt_clean_path = '/home/wwh/Dataset/simulation/segmentation/labels/val/modify'
    gt_file = os.path.join(gt_path, "{:04d}.txt".format(ind))
    gt_clean_file = os.path.join(gt_clean_path, "{:04d}.txt".format(ind))
    with open(gt_clean_file, 'r') as file:
        subset_lines = file.readlines()

    with open(gt_file, 'r') as file:
        full_lines = file.readlines()

    # 存储匹配行的位置
    positions = []

    # 对于子集中的每一行，找到它在完整文件中的位置
    for sub_line in subset_lines:
        index = 0
        for i, full_line in enumerate(full_lines):
            if is_similar(sub_line, full_line, tolerance):
                positions.append(i)
                index = 1
                break
        if index == 0:
            positions.append(-1)

    print('{} anchors are not matched'.format(positions.count(-1)))
    # 去除列表中的相同元素
    positions = list(set(positions))
    return positions


if __name__ == '__main__':
    find_clean_gt(ind=107)
