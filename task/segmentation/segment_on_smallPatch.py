import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
import h5py
import numpy as np
import os
import tifffile as tiff
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from task.pipeline import find_clean_gt
from boundary_loss import dist_map_transform, MyLoss
from sklearn.cluster import KMeans


"""
改进：
1.在新标注的GT上训练，促进边界高频信息的学习，并增加数据量。
2.使用L1 loss，减少边界的模糊。
3.加入抖动，增加数据量的同时，为检测结果提供鲁棒性。
4.由于输入特征和输出GT大同小异，考虑使用对比损失，或一些避免过拟合的手段，如dropout或输入挖空。
5.考虑使用图网络
"""
class dataset(Dataset):
    def __init__(self, data_names, mat_names, debug=True):
        path = '/home/wwh/Dataset/simulation/segmentation/'
        self.disttransform = dist_map_transform(1, 2)
        self.target_size = (22, 22)

        self.movies = []
        self.masks = []
        self.edges = []
        self.max_size = 0
        self.random_move = []

        for mat_name, data_name in zip(mat_names, data_names):
            mat_path = path + mat_name + '.mat'
            mov_path = path + data_name + '.tif'
            mov = np.array(tiff.imread(mov_path))

            with h5py.File(mat_path, 'r') as f:
                anchors = np.array(f['anchors']) - 1  # [x1, y1, x2, y2]
                anchors[2, :] = anchors[2, :] - anchors[0, :]
                anchors[3, :] = anchors[3, :] - anchors[1, :]  # [x, y, w, h]
                anchors = anchors.transpose(1, 0)
                num = len(f['masks'][0])
                masks = []
                edges = []
                for i in range(num):
                    masks.append(np.array(f[f['masks'][0][i]][()]) - 1)
                    edges.append(np.array(f[f['edges'][0][i]][()]) - 1)

            # 读取清洗后的 yolo GT列表
            gt_list = find_clean_gt(train=True, ind=int(data_name.split('_')[-1]) - 1)

            anchors = [anchors[i] for i in gt_list]
            masks = [masks[i] for i in gt_list]
            edges = [edges[i] for i in gt_list]

            if debug:
                ind = 11
                mean_img = np.mean(mov, axis=0)
                mask_x, mask_y = zip(*masks[ind].transpose(1, 0))
                edge_y, edge_x = zip(*edges[ind].transpose(1, 0))
                fig, ax = plt.subplots(1)
                ax.imshow(mean_img, cmap='gray')
                ax.scatter(mask_x, mask_y, s=1, c='blue', alpha=0.5)
                ax.scatter(edge_x, edge_y, s=1, c='green', alpha=0.5)
                rect = patches.Rectangle((max(anchors[ind][0] - 1, 0), max(anchors[ind][1] - 1, 0)),
                                         anchors[ind][2] + 2, anchors[ind][3] + 2,
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.show()

            for i in range(len(anchors)):
                x1, y1, w, h = anchors[i]
                if w == 0 or h == 0:
                    continue
                x1, y1, w, h = max(x1 - 2, 0), max(y1 - 2, 0), min(w + 4, mov.shape[-1]), min(h + 4, mov.shape[-1])
                self.max_size = max(self.max_size, w, h)
                # 对x1,y1的坐标位置做一个小的随机扰动
                x_, y_ = np.random.randint(-1, 1), np.random.randint(-1, 1)
                self.random_move.append([x_, y_])
                x1, y1 = max(x1 + x_, 0), max(y1 + y_, 0)
                x2, y2 = min(x1 + w, mov.shape[-1]), min(y1 + h, mov.shape[-1])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                self.movies.append(mov[:, y1:y2, x1:x2])

                temp_mask = np.copy(masks[i].transpose(1, 0))
                temp_mask[:, 0] = temp_mask[:, 0] - x1
                temp_mask[:, 1] = temp_mask[:, 1] - y1
                self.masks.append(temp_mask)

                temp_edge = np.copy(edges[i].transpose(1, 0))
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

    def __getitem__(self, index, debug=True, cluster=True):
        movie = self.movies[index]
        mask = self.masks[index].astype(np.int32)
        edge = self.edges[index].astype(np.int32)
        mask_img = np.zeros((movie.shape[1], movie.shape[2]))
        mask_img[mask[:, 1], mask[:, 0]] = 1
        edge_img = np.zeros((movie.shape[1], movie.shape[2]))
        edge_img[edge[:, 1], edge[:, 0]] = 1
        pad_movie, pad_edge = self.pad_movie_label(index, movie, mask_img, edge_img, self.target_size)
        corr_matrix = self.corr_mov(pad_movie)
        if debug:
            mean_img = np.mean(pad_movie, axis=0)
            fig, ax = plt.subplots(1)
            ax.imshow(mean_img, cmap='gray')
            ax.imshow(pad_edge, alpha=0.5)
            plt.show()
        # if cluster:
        #     X = self.corr_mov(movie).reshape(-1, movie.shape[1]*movie.shape[2])
        #     center_pixel_feature = X[:, round(movie.shape[1]/2)*movie.shape[2] + round(movie.shape[2]/2)]
        #
        #     # 步骤2: 计算最远的像素点
        #     distances = np.linalg.norm(X - center_pixel_feature[:, np.newaxis], axis=0)
        #     farthest_pixel_index = np.argmax(distances)
        #
        #     # 步骤3: 设置初始聚类中心
        #     init_centers = np.array([center_pixel_feature, X[:, farthest_pixel_index]])
        #
        #     # 步骤4: 应用K-means聚类，使用自定义的初始聚类中心
        #     kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1)
        #     kmeans = kmeans.fit(X)
        #     labels = kmeans.labels_
        #
        #     # 将聚类结果重塑回原始图像尺寸以便可视化
        #     clustered_img = labels.reshape(movie.shape[1], movie.shape[2])
        #
        #     # 可视化聚类结果
        #     plt.imshow(clustered_img, cmap='viridis')
        #     plt.title('K-means Clustering Results')
        #     plt.show()

        corr_matrix = torch.from_numpy(np.expand_dims(corr_matrix, 0).astype(np.float32))
        pad_edge = torch.from_numpy(pad_edge.astype(np.float32))
        # 建立水平集
        dist_map = self.disttransform(pad_edge)[1]

        return corr_matrix, (pad_edge, dist_map)

    def __len__(self):
        return len(self.movies)

    def pad_movie_label(self, index, movie, mask_img, edge_img, target_size=(22, 22)):
        pad_y = max(target_size[0] - movie.shape[1], 0)
        pad_x = max(target_size[1] - movie.shape[2], 0)
        pad_y_l, pad_y_r = pad_y // 2, pad_y - pad_y // 2
        pad_x_l, pad_x_r = pad_x // 2, pad_x - pad_x // 2

        pad_movie = np.pad(movie, ((0, 0), (pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
        pad_mask = np.pad(mask_img, ((pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)
        pad_edge = np.pad(edge_img, ((pad_y_l, pad_y_r), (pad_x_l, pad_x_r)), 'constant', constant_values=0)

        x_, y_ = self.random_move[index]
        if pad_movie.shape[1] > self.target_size[0]:
            if y_ > 0:
                pad_movie = pad_movie[:, :self.target_size[0], :]
                pad_mask = pad_mask[:self.target_size[0], :]
                pad_edge = pad_edge[:self.target_size[0], :]
            else:
                pad_movie = pad_movie[:, -self.target_size[0]:, :]
                pad_mask = pad_mask[-self.target_size[0]:, :]
                pad_edge = pad_edge[-self.target_size[0]:, :]
        if pad_movie.shape[2] > self.target_size[1]:
            if x_ > 0:
                pad_movie = pad_movie[:, :, :self.target_size[1]]
                pad_mask = pad_mask[:, :self.target_size[1]]
                pad_edge = pad_edge[:, :self.target_size[1]]
            else:
                pad_movie = pad_movie[:, :, -self.target_size[1]:]
                pad_mask = pad_mask[:, -self.target_size[1]:]
                pad_edge = pad_edge[:, -self.target_size[1]:]

        blurred_pad_edge = gaussian_filter(pad_edge, sigma=1)
        blurred_pad_edge = blurred_pad_edge / np.max(blurred_pad_edge)
        blurred_pad_edge[pad_edge == 1] = 1

        blurred_pad_mask = pad_mask * blurred_pad_edge
        class_vector = blurred_pad_mask.flatten()
        # pad_edge = pad_edge.flatten()
        return pad_movie, pad_edge

    def corr_matrix(self, movie):
        movie = movie.reshape(movie.shape[0], -1).transpose(1, 0)
        # 加上非常小的随机噪声，使计算相关系数正常进行
        for i in range(movie.shape[0]):
            if np.sum(movie[i, :]) == 0:
                movie[i, :] = np.random.rand(movie.shape[1]) * 1e-5
        matrix = np.corrcoef(movie)
        return matrix

    def corr_mov(self, movie):
        t, h, w = movie.shape
        movie = movie.reshape(t, -1)
        # 加上非常小的随机噪声，使计算相关系数正常进行
        for i in range(h*w):
            if np.sum(movie[:, i]) == 0:
                movie[:, i] = np.random.rand(t) * 1e-5

        correlation_matrix = np.corrcoef(movie, rowvar=False)
        return correlation_matrix.reshape(h*w, h, w)


class Model(nn.Module):
    def __init__(self, in_channels=1, out_channels=484):
        super(Model, self).__init__()
        resnet18 = models.resnet18()

        # 使用ResNet18的原始第一层，以适应1通道的输入
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        # ResNet层
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        # Global Average Pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 输出层
        self.fc = nn.Linear(512, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)


def dice_value(pred, target):
    edge, dist_map = target
    pred = pred.cpu().detach().numpy()
    target = edge.cpu().detach().numpy()
    iflat = pred.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    return ((2. * intersection) / (iflat.sum() + tflat.sum()))


if __name__ == "__main__":
    ''' <<<<<<<<<<< setup >>>>>>>>>>> '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    ''' <<<<<<<<<<< load dataset >>>>>>>>>>> '''
    dataset = dataset(data_names=['raw_20um_1'],
                      mat_names=['segmentation_20um_1'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    ''' <<<<<<<<<<< dataloader >>>>>>>>>>> '''
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    ''' <<<<<<<<<<< network >>>>>>>>>>> '''
    network = Model()
    checkpoint = None
    ''' <<<<<<<<<<< loss >>>>>>>>>>> '''
    loss_fn = MyLoss()

    ''' <<<<<<<<<<< optimizer and lr_scheduler >>>>>>>>>>> '''
    lr = 0.0001
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    n_epochs = 100
    total_steps = n_epochs * len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=5 * lr, total_steps=total_steps)

    ''' <<<<<<<<<<< seed >>>>>>>>>>> '''
    from accelerate.utils import set_seed

    set_seed(42)

    ''' <<<<<<<<<<< metrics >>>>>>>>>>> '''
    metrics = {"dice": dice_value}

    ''' <<<<<<<<<<< callback >>>>>>>>>>> '''
    from framework.common_callback import EpochCheckpoint, EpochTrainLog, EpochEvalLog

    log_folder = "/home/wwh/code/DLT-main/log/segmentation/resnet18/exp11_noise_edge_boundary_focal_loss_shake_filter_unet/"
    weights_folder = os.path.join(log_folder, "weights")
    os.makedirs(weights_folder, exist_ok=True)
    callbacks = [
        EpochCheckpoint(weights_folder, save_freq=1),
        EpochTrainLog(log_folder, save_freq=1),
        EpochEvalLog(log_folder, save_freq=1),
    ]

    from framework.model import Model

    model = Model(network, loss_fn, optimizer, metrics_dict=metrics, lr_scheduler=lr_scheduler)
    if checkpoint is not None:
        print('load checkpoint from %s' % checkpoint)
        model.load_ckpt(checkpoint)

    history = model.fit(train_loader, val_loader, epochs=n_epochs,
                        ckpt_path=os.path.join(weights_folder, "checkpoint.pth"),
                        callbacks=callbacks,
                        early_stopping=True,
                        patience=200,
                        monitor="val_dice",
                        mode="max")