from pathlib import Path
from itertools import repeat
from operator import itemgetter, mul
from functools import partial, reduce
from multiprocessing import cpu_count
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import map_, class2one_hot, one_hot2dist, id_, simplex
from utils import one_hot, depth
from typing import List, cast

from torch import Tensor, einsum
from torch import nn
import torch.nn.functional as F


F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss, self).__init__()

    def forward(self, probs, dist_maps):
        multipled = einsum("bp,bp->bp", probs.type(torch.float32), dist_maps.type(torch.float32))
        loss = multipled.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, p, targets):

        # 对于真实标签为1的情况，计算p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # 计算每个类别的alpha值
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # 计算二元交叉熵损失
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))

        # 计算Focal Loss的修正项
        focal_modulation = (1 - p_t) ** self.gamma

        # 应用Focal Loss的公式
        focal_loss = alpha_factor * focal_modulation * bce

        return focal_loss.mean()



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        return self.criterion(iflat, tflat)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits.view(-1))
        target_probs = F.softmax(targets.view(-1))
        return self.criterion(log_probs, target_probs)


class MyLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(MyLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.surface_loss = SurfaceLoss()
        self.focal_loss = FocalLoss(alpha=0.8, gamma=1.5)

    def forward(self, pred, target):
        edge, dist_map = target
        dice_loss = self.dice_loss(pred, edge)
        surface_loss = self.surface_loss(pred, dist_map)
        focal_loss = self.focal_loss(pred, edge)
        return self.alpha * surface_loss + (1 - self.alpha) * dice_loss + focal_loss

