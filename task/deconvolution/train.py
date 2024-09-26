from dataset import dataset
from model import Deconvolution
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import numpy as np
import os


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    ''' <<<<<<<<<<< load dataset >>>>>>>>>>> '''
    dataset = dataset(data_name='clean_20um_89', mat_name='segmentation_20um_89')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    ''' <<<<<<<<<<< dataloader >>>>>>>>>>> '''
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    ''' <<<<<<<<<<< network >>>>>>>>>>> '''
    network = Deconvolution()

    ''' <<<<<<<<<<< loss >>>>>>>>>>> '''
    loss_fn = Loss()

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
    metrics = {"corr": corr_value}

    ''' <<<<<<<<<<< callback >>>>>>>>>>> '''
    from framework.common_callback import EpochCheckpoint, EpochTrainLog, EpochEvalLog

    log_folder = "/home/wwh/code/DLT-main/log/deconvolution/exp1_baseline"
    weights_folder = os.path.join(log_folder, "weights")
    os.makedirs(weights_folder, exist_ok=True)
    callbacks = [
        EpochCheckpoint(weights_folder, save_freq=1),
        EpochTrainLog(log_folder, save_freq=1),
        EpochEvalLog(log_folder, save_freq=1),
    ]

    from framework.model import Model

    model = Model(network, loss_fn, optimizer, metrics_dict=metrics, lr_scheduler=lr_scheduler)
    history = model.fit(train_loader, val_loader, epochs=n_epochs,
                        ckpt_path=os.path.join(weights_folder, "checkpoint.pth"),
                        callbacks=callbacks,
                        early_stopping=True,
                        patience=90,
                        monitor="val_corr",
                        mode="max")
