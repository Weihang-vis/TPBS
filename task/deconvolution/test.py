from dataset import dataset
from model import Deconvolution
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
from tqdm import tqdm
import sys


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


def corr_value(pred, target):
    """
    :param pred: [b, 1, 1000]
    :param target: [b, 1, 1000]
    :return:  average correlation value
    """
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    corr = 0
    for i in range(pred.shape[0]):
        corr += np.corrcoef(pred[i, 0, :], target[i, 0, :])[0, 1]
    return corr / pred.shape[0]


if __name__ == "__main__":
    ''' <<<<<<<<<<< setup >>>>>>>>>>> '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ''' <<<<<<<<<<< load dataset >>>>>>>>>>> '''
    test_set = dataset(data_name='clean_20um_89', mat_name='segmentation_20um_89')

    ''' <<<<<<<<<<< dataloader >>>>>>>>>>> '''
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    ''' <<<<<<<<<<< network >>>>>>>>>>> '''
    pth_path = '/home/wwh/code/DLT-main/log/deconvolution/exp1_baseline/weights/0063_of_0100_checkpoint.pth'
    network = Deconvolution()
    network.load_state_dict(torch.load(pth_path, map_location="cpu"))
    network.cuda()

    log_folder = "/home/wwh/code/DLT-main/log/deconvolution/exp1_baseline/test"
    os.makedirs(log_folder, exist_ok=True)

    loop = tqdm(enumerate(test_loader, start=1), total=len(test_loader), file=sys.stdout)
    corr_values = []
    for step, batch in loop:
        (pad_movie, pad_mask), signal = batch
        pad_movie = pad_movie.cuda()
        pad_mask = pad_mask.cuda()
        signal = signal.cuda()

        network.eval()
        with torch.no_grad():
            output = network((pad_movie, pad_mask))
            corr = corr_value(output, signal)
            corr_values.append(corr)
        # 将output和signal，以及mask覆盖像素的原始平均信号可视化
        output = output.cpu().detach().numpy().squeeze()
        signal = signal.cpu().detach().numpy().squeeze()
        pad_mask = pad_mask.cpu().detach().numpy().squeeze()
        pad_movie = pad_movie.cpu().detach().numpy().squeeze()
        raw_signal = np.mean(pad_movie * pad_mask, axis=(1, 2))

        import matplotlib.pyplot as plt
        plt.plot(output, label='output')
        plt.plot(signal, label='gt_signal')
        plt.plot(raw_signal, label='raw_signal')
        plt.legend()
        plt.show()

    mean_corr = np.mean(corr_values)
    print(f"mean corr: {mean_corr}")