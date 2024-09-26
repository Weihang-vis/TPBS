import torch
from torch.utils.data import Dataset, random_split, DataLoader
import h5py
import numpy as np
import os
import tifffile as tiff
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class dataset(Dataset):
    def __init__(self, data_name, mat_name, debug=False):
        path = '/home/wwh/Dataset/simulation/segmentation/'
        self.debug = debug
        self.data_name = data_name
        self.mat_name = mat_name
        self.mat_path = path + self.mat_name + '.mat'
        self.mov_path = path + self.data_name + '.tif'
        self.mov = np.array(tiff.imread(self.mov_path))
        self.mov = self.preprocess()
        self.anno = {}
        self.target_size = (20, 20)
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

            self.anno['signals'] = np.array(f['neur_act'][()]).transpose(1, 0)
            self.anno['idList'] = f['idList'][()].squeeze() - 1
        valid_idx = self.calculate_iou()

        self.movies = []
        self.masks = []
        self.edges = []
        self.signals = []
        self.random_move = []
        for i in range(len(self.anno['anchors'])):
            x1, y1, w, h = self.anno['anchors'][i]
            id = self.anno['idList'][i]
            if w == 0 or h == 0 or valid_idx[i] == 0:
                continue
            x1, y1, w, h = max(x1 - 2, 0), max(y1 - 2, 0), min(w + 4, self.mov.shape[-1]), min(h + 4, self.mov.shape[-1])
            # 对x1,y1的坐标位置做一个小的随机扰动
            x_, y_ = np.random.randint(-2, 2), np.random.randint(-2, 2)
            self.random_move.append([x_, y_])
            x1, y1 = max(x1 + x_, 0), max(y1 + y_, 0)
            x2, y2 = min(x1 + w, self.mov.shape[-1]), min(y1 + h, self.mov.shape[-1])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            self.movies.append(self.mov[:, y1:y2, x1:x2])

            temp_mask = np.copy(self.anno['masks'][i].transpose(1, 0))
            temp_mask[:, 0] = temp_mask[:, 0] - x1
            temp_mask[:, 1] = temp_mask[:, 1] - y1
            # 过滤掉超出边界框的mask
            temp_mask = temp_mask[np.where((temp_mask[:, 0] >= 0) & (temp_mask[:, 0] < w) &
                                           (temp_mask[:, 1] >= 0) & (temp_mask[:, 1] < h))]
            self.masks.append(temp_mask)

            temp_edge = np.copy(self.anno['edges'][i].transpose(1, 0))
            temp_edge[:, 0] = temp_edge[:, 0] - y1
            temp_edge[:, 1] = temp_edge[:, 1] - x1
            temp = np.copy(temp_edge[:, 0])
            temp_edge[:, 0] = temp_edge[:, 1]
            temp_edge[:, 1] = temp
            # 过滤掉超出边界框的edge
            temp_edge = temp_edge[np.where((temp_edge[:, 0] >= 0) & (temp_edge[:, 0] < w) &
                                           (temp_edge[:, 1] >= 0) & (temp_edge[:, 1] < h))]
            self.edges.append(temp_edge)
            self.signals.append(self.anno['signals'][int(id)])

        # standardize
        self.signals = np.array(self.signals)
        self.signals = (self.signals - np.mean(self.signals)) / np.std(self.signals)

        if self.debug:
            idx = 10
            mean_img = np.mean(self.movies[idx], axis=0)
            mask_x, mask_y = zip(*self.masks[idx])
            edge_x, edge_y = zip(*self.edges[idx])
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(mean_img, cmap='gray')
            ax[0].scatter(mask_x, mask_y, s=1, c='blue', alpha=0.5)
            ax[0].scatter(edge_x, edge_y, s=1, c='green', alpha=0.5)
            signal_raw = self.movies[idx][:, list(map(int, mask_y)), list(map(int, mask_x))]
            signal_raw = np.mean(signal_raw, axis=1)
            signal_clean = self.signals[idx]
            ax[1].plot(signal_raw, c='blue')
            ax[1].plot(signal_clean, c='green')
            plt.show()

    def __getitem__(self, index):
        movie = self.movies[index]
        mask = self.masks[index].astype(np.int32)
        mask_img = np.zeros((movie.shape[1], movie.shape[2]))
        mask_img[mask[:, 1], mask[:, 0]] = 1
        pad_movie, pad_mask = self.pad_movie_mask(movie, mask_img, index)
        signal = self.signals[index]

        if self.debug:
            img = np.mean(pad_movie, axis=0)
            mask_y, mask_x = np.where(pad_mask == 1)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img, cmap='gray')
            ax[0].scatter(mask_x, mask_y, s=1, c='blue', alpha=0.5)
            signal_raw = pad_movie[:, list(map(int, mask_y)), list(map(int, mask_x))]
            signal_raw = np.mean(signal_raw, axis=1)
            ax[1].plot(signal_raw, c='blue')
            ax[1].plot(signal, c='green')
            plt.show()

        pad_movie = torch.from_numpy(np.expand_dims(pad_movie, 0).astype(np.float32))
        pad_mask = torch.from_numpy(np.expand_dims(pad_mask, 0).astype(np.float32))
        signal = torch.from_numpy(np.expand_dims(signal, 0).astype(np.float32))

        return (pad_movie, pad_mask), signal

    def __len__(self):
        return len(self.movies)

    def pad_movie_mask(self, movie, mask, index):

        pad_y = max(self.target_size[0] - movie.shape[1], 0)
        pad_x = max(self.target_size[1] - movie.shape[2], 0)

        pad_y_l, pad_y_r = pad_y // 2, pad_y - pad_y // 2
        pad_x_l, pad_x_r = pad_x // 2, pad_x - pad_x // 2

        pad_movie = np.pad(movie, ((0, 0), (pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
        pad_mask = np.pad(mask, ((pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)

        x_, y_ = self.random_move[index]
        if pad_movie.shape[1] > self.target_size[0]:
            if y_ > 0:
                pad_movie = pad_movie[:, :self.target_size[0], :]
                pad_mask = pad_mask[:self.target_size[0], :]
            else:
                pad_movie = pad_movie[:, -self.target_size[0]:, :]
                pad_mask = pad_mask[-self.target_size[0]:, :]
        if pad_movie.shape[2] > self.target_size[1]:
            if x_ > 0:
                pad_movie = pad_movie[:, :, :self.target_size[1]]
                pad_mask = pad_mask[:, :self.target_size[1]]
            else:
                pad_movie = pad_movie[:, :, -self.target_size[1]:]
                pad_mask = pad_mask[:, -self.target_size[1]:]

        return pad_movie, pad_mask

    def preprocess(self, percentile=99.9, bit=12):
        max_value = np.percentile(self.mov, percentile)
        self.mov[self.mov < 0] = 0
        self.mov[self.mov > max_value] = max_value
        self.mov = self.mov / max_value * (2 ** bit - 1)
        # standardization
        self.mov = (self.mov - np.mean(self.mov)) / np.std(self.mov)

        # logarithmic transformation
        # x = np.log2(self.mov + 1)
        return self.mov

    def calculate_iou(self, threshold=0.5):
        N = len(self.anno['anchors'])
        A = np.ones(N)  # 初始化全1数组

        # 计算每个anchor的坐标范围
        x1 = self.anno['anchors'][:, 0]
        y1 = self.anno['anchors'][:, 1]
        x2 = self.anno['anchors'][:, 0] + self.anno['anchors'][:, 2]
        y2 = self.anno['anchors'][:, 1] + self.anno['anchors'][:, 3]

        # 计算每两个anchor之间的交集
        x1_matrix = np.maximum(x1[:, None], x1)
        y1_matrix = np.maximum(y1[:, None], y1)
        x2_matrix = np.minimum(x2[:, None], x2)
        y2_matrix = np.minimum(y2[:, None], y2)

        # 计算交集的面积
        intersection = np.maximum(0, x2_matrix - x1_matrix) * np.maximum(0, y2_matrix - y1_matrix)

        # 计算每个anchor的面积
        area = (x2 - x1) * (y2 - y1)

        # 计算IoU
        iou_matrix = intersection / (area[:, None] + area - intersection + 0.00001)

        # 根据阈值更新数组A
        for i in range(N):
            for j in range(i + 1, N):
                if iou_matrix[i, j] > threshold:
                    A[i] = A[j] = 0

        return A


if __name__ == "__main__":
    dataset = dataset(data_name='clean_20um_89', mat_name='segmentation_20um_89')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)