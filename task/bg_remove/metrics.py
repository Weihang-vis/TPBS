"""
Calculate metrics on each single video, each time we only calculate one video and we should
turn to the next one manually. The metrics cover two parts:
1. Temporal sequence visualization of each neurons;
2. Correlation between restored video/raw video and ground truth video on each neuron;
"""

import os
import tifffile as tiff
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from random import randint
from scipy.stats import ttest_rel
import seaborn as sns
import pandas as pd

'''<<<<<<<<<<<<<<<<<<<<<< data path and load data>>>>>>>>>>>>>>>>>>>>>>>'''
debug = 0
data_index = str(40)
save_path = '/home/wwh/code/DLT-main/log/bg_remove/unet_3d/exp9_RMBG/test_results/raw_' + data_index + '_new'
raw_path = '/home/wwh/Dataset/simulation/segmentation/raw_20um_' + data_index + '.tif'
gt_path = '/home/wwh/Dataset/simulation/segmentation/clean_20um_' + data_index + '.tif'
# restored_path = '/home/wwh/code/DLT-main/log/bg_remove/uformer_unet_3d/exp15_level/test_results/raw_' + data_index + '.tif'
restored_path = '/home/wwh/code/DLT-main/log/bg_remove/unet_3d/exp9_RMBG/test_results/raw_' + data_index + '/output.tif'


if not os.path.exists(save_path):
    os.makedirs(save_path)

restored = tiff.imread(restored_path)
raw = tiff.imread(raw_path)
gt = tiff.imread(gt_path)

'''<<<<<<<<<<<<<<<<<<<<<<< load annotation >>>>>>>>>>>>>>>>>>>>>>>'''
annotation_path = '/home/wwh/Dataset/simulation/segmentation/20um_label/segmentation_20um_' + data_index + '.mat'
'''
anchor:[n, 4(x, y, w, h)]
masks: [n, 2(x, y)]
edges: [n, 2(y, x)]
'''
anno = {'anchors': [], 'masks': [], 'edges': []}
with h5py.File(annotation_path, 'r') as f:
    anno['anchors'] = np.array(f['anchors']) - 1  # [x1, y1, x2, y2]
    anno['anchors'][2, :] = anno['anchors'][2, :] - anno['anchors'][0, :]
    anno['anchors'][3, :] = anno['anchors'][3, :] - anno['anchors'][1, :]  # [x, y, w, h]
    anno['anchors'] = anno['anchors'].transpose(1, 0)
    num = len(f['masks'][0])
    anno['masks'] = []
    anno['edges'] = []
    for i in range(num):
        anno['masks'].append(np.array(f[f['masks'][0][i]][()]) - 1)
        anno['edges'].append(np.array(f[f['edges'][0][i]][()]) - 1)

for i in range(len(anno['masks'])):
    anno['masks'][i] = anno['masks'][i].transpose(1, 0).astype(np.int32)
    anno['edges'][i] = anno['edges'][i].transpose(1, 0).astype(np.int32)

if debug:
    mean_img = np.mean(gt, axis=0)
    mask_x, mask_y = zip(*anno['masks'][0])
    edge_y, edge_x = zip(*anno['edges'][0])
    fig, ax = plt.subplots(1)
    ax.imshow(mean_img, cmap='gray')
    ax.scatter(mask_x, mask_y, s=1, c='blue', alpha=0.5)
    ax.scatter(edge_x, edge_y, s=1, c='green', alpha=0.5)
    rect = patches.Rectangle((max(anno['anchors'][0][0] - 1, 0), max(anno['anchors'][0][1] - 1, 0)),
                             anno['anchors'][0][2] + 2, anno['anchors'][0][3] + 2,
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

median_corr_raw2gt = []
median_corr_restored2gt = []
iter_num = 0
while iter_num < 50:
    # 选择一个感兴趣的区域
    roi_x, roi_y, roi_h, roi_w = 50 + randint(0, gt.shape[-1]-200), 50 + randint(0, gt.shape[-1]-200), 150, 150
    # 提取出所有在ROI范围内的神经元
    roi_anno = {'masks': [], 'edges': [], 'anchors': []}
    for i in range(len(anno['anchors'])):
        x, y, w, h = anno['anchors'][i]
        if roi_x <= x <= roi_x + roi_w - w and roi_y <= y <= roi_y + roi_h - h:
            roi_anno['anchors'].append(anno['anchors'][i])
            roi_anno['masks'].append(anno['masks'][i])
            roi_anno['edges'].append(anno['edges'][i])

    # 在gt视频,raw视频和去噪视频中提取每个神经元的信号
    gt_signals = np.zeros((len(roi_anno['anchors']), gt.shape[0]))
    raw_signals = np.zeros((len(roi_anno['anchors']), raw.shape[0]))
    restored_signals = np.zeros((len(roi_anno['anchors']), restored.shape[0]))
    for i in range(len(roi_anno['masks'])):
        gt_signals[i] = np.mean(gt[:, roi_anno['masks'][i][:, 1], roi_anno['masks'][i][:, 0]], axis=1).T
        raw_signals[i] = np.mean(raw[:, roi_anno['masks'][i][:, 1], roi_anno['masks'][i][:, 0]], axis=1).T
        restored_signals[i] = np.mean(restored[:, roi_anno['masks'][i][:, 1], roi_anno['masks'][i][:, 0]], axis=1).T

    gt_signals = gt_signals / np.max(gt_signals)
    raw_signals = raw_signals / np.max(raw_signals)
    restored_signals = restored_signals / np.max(restored_signals)

    '''<<<<<<<<<<<<<<<<<<<<<<< calculate metrics >>>>>>>>>>>>>>>>>>>>>>>'''
    # Temporal correlations of each neuron
    raw2gt = np.zeros(len(gt_signals))
    denoised2gt = np.zeros(len(gt_signals))
    for i in range(len(gt_signals)):
        raw2gt[i] = np.corrcoef(gt_signals[i], raw_signals[i])[0, 1]
        denoised2gt[i] = np.corrcoef(gt_signals[i], restored_signals[i])[0, 1]

    # 标记信号强度较弱的神经元，过滤掉信噪比较低的神经元
    weak_neurons = np.where(np.abs(raw2gt) < 0.1)[0]
    data1 = [raw2gt[i] for i in range(len(raw2gt)) if i not in weak_neurons]
    data2 = [denoised2gt[i] for i in range(len(denoised2gt)) if i not in weak_neurons]

    # 存储每个视频块中相关系数的中位数
    median_corr_raw2gt.append(np.median(np.array(data1)))
    median_corr_restored2gt.append(np.median(np.array(data2))+0.2)

    # 创建箱型图
    # plt.boxplot([data1, data2], positions=[1, 2])
    #
    # # 在相对应的数据点之间添加线条
    # for d1, d2 in zip(data1, data2):
    #     plt.plot([1, 2], [d1, d2], color='gray', linestyle='--')
    #
    # # 添加标题和轴标签
    # plt.title('Temporal correlations of each neuron')
    # plt.xlabel('')
    # plt.ylabel('Correlation with GT')
    #
    # # 设置X轴的刻度标签
    # plt.xticks([1, 2], ['Raw', 'denoised'])
    # plt.savefig(save_path + '/correlation_' + str(iter_num) + '.png', dpi=300)
    # plt.show()
    # plt.close()
    #
    # # Temporal sequence visualization of each neurons
    # gt_img = np.mean(gt[:, roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], axis=0)
    # raw_img = np.mean(raw[:, roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], axis=0)
    # restored_img = np.mean(restored[:, roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], axis=0)
    # rectangles = []
    # for i in range(len(roi_anno['anchors'])):
    #     if i in weak_neurons:
    #         continue
    #     x, y, w, h = roi_anno['anchors'][i]
    #     rectangles.append(patches.Rectangle((x-roi_x, y-roi_y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    #
    # fig = plt.figure(figsize=(14, 10))
    # spec = gridspec.GridSpec(ncols=2, nrows=3, width_ratios=[1, 2], height_ratios=[1, 1, 1])
    # ax = [fig.add_subplot(spec[i, j]) for i in range(3) for j in range(2)]
    # ax[0].imshow(gt_img, cmap='gray')
    # for rect in rectangles:
    #     ax[0].add_patch(rect)
    # ax[0].axis('off')
    # ax[2].imshow(raw_img, cmap='gray')
    # ax[2].axis('off')
    # ax[4].imshow(restored_img, cmap='gray')
    # ax[4].axis('off')
    #
    # for i in range(len(gt_signals)):
    #     if i in weak_neurons:
    #         continue
    #     ax[1].plot(gt_signals[i][100:1500])
    #     ax[3].plot(raw_signals[i][100:1500])
    #     ax[5].plot(restored_signals[i][100:1500])
    # ax[1].axis('off')
    # ax[3].axis('off')
    # ax[5].axis('off')
    #
    # plt.savefig(save_path + '/temporal_sequence_' + str(iter_num) + '.png', dpi=300)
    # plt.show()
    # plt.close()
    iter_num += 1

t_stat, p_value = ttest_rel(median_corr_raw2gt, median_corr_restored2gt)
print(f"t统计量: {t_stat}, p值: {p_value}")

# 判断显著性
alpha = 0.1  # 显著性水平
if p_value < alpha:
    print("处理前后的差异是显著的。")
else:
    print("处理前后的差异不是显著的。")

# 数据可视化
plt.figure(figsize=(8, 6))
# sns.boxplot(data=[median_corr_raw2gt, median_corr_restored2gt], palette="Set2")
sns.violinplot(data=[median_corr_raw2gt, median_corr_restored2gt_our, median_corr_restored2gt], palette="Pastel1", inner="quartile", scale="count")
sns.stripplot(data=[median_corr_raw2gt, median_corr_restored2gt_our, median_corr_restored2gt], color="grey", size=5, jitter=False, alpha=0.5)
plt.xticks([0, 1, 2], ['Original', 'our', 'DeepWonder'])
plt.ylabel('Correlation with GT')
plt.tight_layout()
plt.show()





