import json
import torch
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader
import h5py
import numpy as np
import os
import tifffile as tiff
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from segment_on_smallPatch import Model, dice_value
from task.segmentation.utils import evaluate
from scipy.ndimage import gaussian_filter


class dataset(Dataset):
    def __init__(self, data_name, mat_name, debug=False):
        path = '/home/wwh/Dataset/simulation/segmentation/'
        self.data_name = data_name
        self.mat_name = mat_name
        self.mat_path = path + self.mat_name + '.mat'
        self.mov_path = path + self.data_name + '.tif'
        self.mov = np.array(tiff.imread(self.mov_path))
        self.anno = {}
        with h5py.File(self.mat_path, 'r') as f:
            self.anno['anchors'] = np.array(f['anchors']) - 1  # [x1, y1, x2, y2]
            self.anno['anchors'][2, :] = self.anno['anchors'][2, :] - self.anno['anchors'][0, :]
            self.anno['anchors'][3, :] = self.anno['anchors'][3, :] - self.anno['anchors'][1, :]  # [x, y, w, h]
            self.anno['anchors'] = self.anno['anchors'].transpose(1, 0)
            num = len(f['masks'][0])
            self.anno['masks'] = []
            self.anno['edges'] = []
            for i in range(num):
                self.anno['masks'].append(np.array(f[f['masks'][0][i]][()]) - 1)
                self.anno['edges'].append(np.array(f[f['edges'][0][i]][()]) - 1)

        if debug:
            index = 9
            mean_img = np.mean(self.mov, axis=0)
            mask_x, mask_y = zip(*self.anno['masks'][index].transpose(1, 0))
            edge_y, edge_x = zip(*self.anno['edges'][index].transpose(1, 0))
            fig, ax = plt.subplots(1)
            ax.imshow(mean_img, cmap='gray')
            ax.scatter(mask_x, mask_y, s=1, c='blue', alpha=0.5)
            ax.scatter(edge_x, edge_y, s=1, c='green', alpha=0.5)
            rect = patches.Rectangle((max(self.anno['anchors'][index][0] - 1, 0), max(self.anno['anchors'][index][1] - 1, 0)),
                                     self.anno['anchors'][index][2] + 2, self.anno['anchors'][index][3] + 2,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()

        self.movies = []
        self.masks = []
        self.edges = []
        self.crop_pos = []
        self.max_size = 0
        for i in range(len(self.anno['anchors'])):
            x1, y1, w, h = self.anno['anchors'][i]
            if w == 0 or h == 0:
                continue
            x1, y1, w, h = max(x1 - 2, 0), max(y1 - 2, 0), min(w + 4, self.mov.shape[-1]), min(h + 4, self.mov.shape[-1])
            self.max_size = max(self.max_size, w, h)
            # # 对x1,y1的坐标位置做一个小的随机扰动
            # x_, y_ = np.random.randint(-1, 1), np.random.randint(-1, 1)
            # x1, y1 = max(x1 + x_, 0), max(y1 + y_, 0)
            x2, y2 = min(x1 + w, self.mov.shape[-1]), min(y1 + h, self.mov.shape[-1])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            self.movies.append(self.mov[:, y1:y2, x1:x2])
            self.crop_pos.append((x1, y1))

            temp_mask = np.copy(self.anno['masks'][i].transpose(1, 0))
            temp_mask[:, 0] = temp_mask[:, 0] - x1
            temp_mask[:, 1] = temp_mask[:, 1] - y1
            self.masks.append(temp_mask)

            temp_edge = np.copy(self.anno['edges'][i].transpose(1, 0))
            temp_edge[:, 0] = temp_edge[:, 0] - y1
            temp_edge[:, 1] = temp_edge[:, 1] - x1
            temp = np.copy(temp_edge[:, 0])
            temp_edge[:, 0] = temp_edge[:, 1]
            temp_edge[:, 1] = temp
            self.edges.append(temp_edge)

        if debug:
            mean_img = np.mean(self.movies[0], axis=0)
            mask_x, mask_y = zip(*self.masks[0])
            edge_x, edge_y = zip(*self.edges[0])
            fig, ax = plt.subplots(1)
            ax.imshow(mean_img, cmap='gray')
            ax.scatter(mask_x, mask_y, s=1, c='blue', alpha=0.5)
            ax.scatter(edge_x, edge_y, s=1, c='green', alpha=0.5)
            plt.show()

        self.target_size = (20, 20)

    def __getitem__(self, index):
        movie = self.movies[index]
        crop_pos = self.crop_pos[index]
        edge = self.edges[index].astype(np.int32)
        mask = self.masks[index].astype(np.int32)

        mask_img = np.zeros((movie.shape[1], movie.shape[2]))
        mask_img[mask[:, 1], mask[:, 0]] = 1
        edge_img = np.zeros((movie.shape[1], movie.shape[2]))
        edge_img[edge[:, 1], edge[:, 0]] = 1

        pad_movie, class_vector = self.pad_movie_label(movie, mask_img, edge_img, self.target_size)
        # pad_movie, class_vector = self.pad_movie_label(movie, mask_img, self.target_size)
        corr_matrix = self.corr_matrix(pad_movie)
        corr_matrix = torch.from_numpy(np.expand_dims(corr_matrix, 0).astype(np.float32))
        class_vector = torch.from_numpy(class_vector.astype(np.float32))

        return corr_matrix, class_vector, movie, crop_pos

    def __len__(self):
        return len(self.movies)

    # def pad_movie_label(self, movie, mask_img, target_size=(20, 20)):
    #     pad_y = max(target_size[0] - movie.shape[1], 0)
    #     pad_x = max(target_size[1] - movie.shape[2], 0)
    #     pad_y_l, pad_y_r = pad_y // 2, pad_y - pad_y // 2
    #     pad_x_l, pad_x_r = pad_x // 2, pad_x - pad_x // 2
    #
    #     pad_movie = np.pad(movie, ((0, 0), (pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
    #     pad_mask = np.pad(mask_img, ((pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
    #     class_vector = pad_mask.flatten()
    #     return pad_movie, class_vector
    def pad_movie_label(self, movie, mask_img, edge_img, target_size=(20, 20)):
        pad_y = max(target_size[0] - movie.shape[1], 0)
        pad_x = max(target_size[1] - movie.shape[2], 0)
        pad_y_l, pad_y_r = pad_y // 2, pad_y - pad_y // 2
        pad_x_l, pad_x_r = pad_x // 2, pad_x - pad_x // 2

        pad_movie = np.pad(movie, ((0, 0), (pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
        pad_mask = np.pad(mask_img, ((pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
        pad_edge = np.pad(edge_img, ((pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)

        blurred_pad_edge = gaussian_filter(pad_edge, sigma=1)
        blurred_pad_edge = blurred_pad_edge / np.max(blurred_pad_edge)
        blurred_pad_edge[pad_edge == 1] = 1

        blurred_pad_mask = pad_mask * blurred_pad_edge
        class_vector = blurred_pad_mask.flatten()
        return pad_movie, pad_edge

    def corr_matrix(self, movie):
        movie = movie.reshape(movie.shape[0], -1).transpose(1, 0)
        # 加上非常小的随机噪声，使计算相关系数正常进行
        for i in range(movie.shape[0]):
            if np.sum(movie[i, :]) == 0:
                movie[i, :] = np.random.rand(movie.shape[1]) * 1e-5
        matrix = np.corrcoef(movie)
        return matrix


if __name__ == "__main__":

    ''' <<<<<<<<<<< setup >>>>>>>>>>> '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    ''' <<<<<<<<<<< load dataset >>>>>>>>>>> '''
    test_set = dataset(data_name='raw_20um_107', mat_name='segmentation_20um_107')

    ''' <<<<<<<<<<< dataloader >>>>>>>>>>> '''
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    ''' <<<<<<<<<<< network >>>>>>>>>>> '''
    pth_path = '/home/wwh/code/DLT-main/log/segmentation/resnet18/exp10_noise_edge_boundary_focal_loss_filter_20/weights/0054_of_0100_checkpoint.pth'
    network = Model(out_channels=400)
    network.load_state_dict(torch.load(pth_path, map_location="cpu"))
    network.cuda()

    log_folder = "/home/wwh/code/DLT-main/log/segmentation/resnet18/exp10_noise_edge_boundary_focal_loss_filter/test"
    os.makedirs(log_folder, exist_ok=True)

    loop = tqdm(enumerate(test_loader, start=1), total=len(test_loader), file=sys.stdout)

    annotations_gt = []
    annotations_pre = []

    for step, batch in loop:
        corr_matrix, class_vector, movie, crop_pos = batch
        corr_matrix = corr_matrix.cuda()
        movie = movie.squeeze().numpy()

        network.eval()
        with torch.no_grad():
            output = network(corr_matrix)
        output = output.cpu().numpy()
        # 将输出的向量转换为图像
        output = output.reshape(20, 20)
        output = output > 0.8
        output = output.astype(np.int32)

        class_vector = class_vector.squeeze().numpy().astype(np.int32)
        # 将图像大小还原为padding之前的尺寸
        pad_y_l = (output.shape[0] - movie.shape[1]) // 2
        pad_y_r = output.shape[0] - movie.shape[1] - pad_y_l
        pad_x_l = (output.shape[1] - movie.shape[2]) // 2
        pad_x_r = output.shape[1] - movie.shape[2] - pad_x_l

        output_ = output[pad_y_l:output.shape[0] - pad_y_r, pad_x_l:output.shape[1] - pad_x_r]
        class_vector_ = class_vector[pad_y_l:output.shape[0] - pad_y_r, pad_x_l:output.shape[1] - pad_x_r]
        # 将mask图像解码成patch中的坐标点
        mask_y_gt, mask_x_gt = np.where(class_vector_ == 1)
        mask_y, mask_x = np.where(output_ == 1)

        # if len(mask_x_gt) > 0:
        #     annotations_gt.append({'coordinates': [[int(x), int(y)] for x, y in
        #                            zip(mask_x_gt + crop_pos[0].numpy(), mask_y_gt + crop_pos[1].numpy())]})
        # if len(mask_x) > 0:
        #     annotations_pre.append({'coordinates': [[int(x), int(y)] for x, y in
        #                            zip(mask_x + crop_pos[0].numpy(), mask_y + crop_pos[1].numpy())]})
        # 绘制图像
        mean_img = np.mean(movie, axis=0)
        fig, ax = plt.subplots(1, figsize=mean_img.shape[::-1])
        ax.imshow(mean_img, cmap='gray')
        ax.scatter(mask_x, mask_y, s=100, c='blue', alpha=0.5)
        ax.scatter(mask_x_gt, mask_y_gt, s=100, c='green', alpha=0.5)
        plt.savefig(os.path.join(log_folder, f"{step}.png"))
        plt.close()

    # # 将annotations保存为neurofinder的json格式
    # annotations_gt_json = json.dumps(annotations_gt)
    # annotations_pre_json = json.dumps(annotations_pre)
    # with open(os.path.join(log_folder, "annotations_gt.json"), 'w') as f:
    #     f.write(annotations_gt_json)
    # with open(os.path.join(log_folder, "annotations_pre.json"), 'w') as f:
    #     f.write(annotations_pre_json)
    #
    # evaluate([log_folder + '/annotations_gt.json', log_folder+ '/annotations_pre.json'])
    #
    # # 将mask绘制在完整图像上
    # background_image_path = '/home/wwh/Dataset/simulation/segmentation/raw_20um_107.tif'
    # background = np.array(tiff.imread(background_image_path))
    # background = np.mean(background, axis=0)
    # background = (background / background.max() * 255).astype(np.uint8)
    # background = Image.fromarray(background).convert('RGBA')
    #
    # bg_width, bg_height = background.size
    #
    # # 初始化gt和pre的mask数组
    # mask_gt = np.zeros((bg_height, bg_width), dtype=np.int32)
    # mask_pre = np.zeros((bg_height, bg_width), dtype=np.int32)
    #
    # # 累加gt和pre的点到对应的mask数组
    # for annotation_gt in annotations_gt:
    #     for x, y in annotation_gt['coordinates']:
    #         mask_gt[y, x] += 1
    # for annotation_pre in annotations_pre:
    #     for x, y in annotation_pre['coordinates']:
    #         mask_pre[y, x] += 1
    #
    # # 归一化mask数组
    # max_val_gt = np.max(mask_gt)
    # max_val_pre = np.max(mask_pre)
    # if max_val_gt > 0:
    #     mask_gt = (mask_gt / max_val_gt * 255).astype(np.uint8)
    # if max_val_pre > 0:
    #     mask_pre = (mask_pre / max_val_pre * 255).astype(np.uint8)
    #
    # # 创建彩色透明图层
    # overlay_gt = Image.fromarray(
    #     np.stack([np.ones_like(mask_gt)*255, np.zeros_like(mask_gt), np.zeros_like(mask_gt),  mask_gt], axis=-1), 'RGBA')
    # overlay_pre = Image.fromarray(
    #     np.stack([np.zeros_like(mask_pre), np.ones_like(mask_pre)*255, np.zeros_like(mask_pre), mask_pre], axis=-1), 'RGBA')
    #
    # # 将彩色透明图层叠加到背景图像上
    # combined = Image.alpha_composite(background, overlay_gt)
    # combined = Image.alpha_composite(combined, overlay_pre)
    #
    # # 显示或保存最终图像
    # combined.show()
    # combined.save(os.path.join(log_folder, f"all.png"))





