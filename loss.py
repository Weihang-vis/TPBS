import torch
import cv2
import torch.nn.functional as F


class Lloss(torch.nn.Module):
    def __init__(self, index=0.7, mu=0.1, sigma=0.9, weight=False):
        super(Lloss, self).__init__()
        self.L1loss = torch.nn.L1Loss()
        self.L2loss = torch.nn.MSELoss()
        self.index = index
        self.mu = mu
        self.sigma = sigma
        self.weight = weight

    def forward(self, predictions, targets):
        if len(predictions) == 1:
            if self.weight:
                weights = torch.exp(-((targets - self.mu) ** 2) / (2 * self.sigma ** 2))
                weighted_targets = weights * targets
                weighted_predictions = weights * predictions[0]
            else:
                weighted_targets = targets
                weighted_predictions = predictions[0]
            L1_loss = self.L1loss(weighted_predictions, weighted_targets)
            L2_loss = self.L2loss(weighted_predictions, weighted_targets)
            total_loss = self.index * L1_loss + (1 - self.index) * L2_loss
            return total_loss
        if len(predictions) == 2:
            if self.weight:
                weights = torch.exp(-((targets - self.mu) ** 2) / (2 * self.sigma ** 2))
                weighted_targets_mov = weights * targets
                weighted_predictions_mov = weights * predictions[0]
            else:
                weighted_targets_mov = targets
                weighted_predictions_mov = predictions[0]
            L1_loss_mov = self.L1loss(weighted_predictions_mov, weighted_targets_mov)
            L2_loss_mov = self.L2loss(weighted_predictions_mov, weighted_targets_mov)

            mean_target = torch.mean(targets, dim=2, keepdim=False)
            mean_prediction = predictions[1]
            if self.weight:
                weights = torch.exp(-((mean_target - self.mu) ** 2) / (2 * self.sigma ** 2))
                weighted_target_mean = weights * mean_target
                weighted_prediction_mean = weights * mean_prediction
            else:
                weighted_target_mean = mean_target
                weighted_prediction_mean = mean_prediction
            L1_loss_mean = self.L1loss(weighted_prediction_mean, weighted_target_mean)
            L2_loss_mean = self.L2loss(weighted_prediction_mean, weighted_target_mean)

            total_loss = self.index * (L1_loss_mov + L1_loss_mean) + (1 - self.index) * (L2_loss_mov + L2_loss_mean)
            return total_loss


class SsimLoss(torch.nn.Module):
    def __init__(self, window_size=11, k1=0.01, k2=0.03):
        super(SsimLoss, self).__init__()
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.C1 = self.k1 ** 2
        self.C2 = self.k2 ** 2

    def forward(self, img1, img2):
        if len(img1) == 1:
            return self._ssim_3d(img1[0], img2)
        if len(img1) == 2:
            return self._ssim_3d(img1[0], img2) + self._ssim_2d(img1[1], img2)

    def _ssim_3d(self, img1, img2):
        window = torch.tensor(cv2.getGaussianKernel(self.window_size, 1.5), dtype=img1.dtype, device=img1.device)
        window = window @ window.T
        window = window.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        mu1 = F.conv3d(img1, window, padding=(0, self.window_size // 2, self.window_size // 2), groups=img1.shape[1])
        mu2 = F.conv3d(img2, window, padding=(0, self.window_size // 2, self.window_size // 2), groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 ** 2, window, padding=(0, self.window_size // 2, self.window_size // 2),
                             groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv3d(img2 ** 2, window, padding=(0, self.window_size // 2, self.window_size // 2),
                             groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=(0, self.window_size // 2, self.window_size // 2),
                           groups=img1.shape[1]) - mu1_mu2

        luminance = (2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)
        contrast = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        structure = (sigma12 + self.C2) / (sigma1_sq.sqrt() * sigma2_sq.sqrt() + self.C2)

        ssim = luminance * contrast * structure

        return 1 - ssim.mean()

    def _ssim_2d(self, img1, img2):
        img2 = torch.mean(img2, dim=2, keepdim=False)

        window = torch.tensor(cv2.getGaussianKernel(self.window_size, 1.5), dtype=img1.dtype, device=img1.device)
        window = window @ window.T
        window = window.unsqueeze(0).unsqueeze(0)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=self.window_size // 2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=self.window_size // 2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=img1.shape[1]) - mu1_mu2

        luminance = (2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)
        contrast = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        structure = (sigma12 + self.C2) / (sigma1_sq.sqrt() * sigma2_sq.sqrt() + self.C2)

        ssim = luminance * contrast * structure

        return 1 - ssim.mean()


class TemporalLoss(torch.nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()

    def forward(self, img1, img2):
        temporal_loss = torch.mean(torch.abs((img2[:, :, 1:, :, :] - img2[:, :, :-1, :, :] -
                                   (img1[:, :, 1:, :, :] - img1[:, :, :-1, :, :]))))
        return temporal_loss


class MyLoss(torch.nn.Module):
    def __init__(self, alpha=1, mu=0.4, sigma=6, weight=False):
        super(MyLoss, self).__init__()
        self.ssim_loss = SsimLoss()
        self.l_loss = Lloss(mu=mu, sigma=sigma, weight=weight)
        self.tem_loss = TemporalLoss()
        self.alpha = alpha

    def forward(self, img1, img2):
        # return self.l_loss(img1, img2) + self.alpha * self.ssim_loss(img1, img2) + \
        #        (1 - self.alpha) * self.tem_loss(img1[0], img2)
        # return self.l_loss(img1, img2) + self.alpha * self.tem_loss(img1[0], img2)
        return self.l_loss(img1, img2)



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    loss1 = SsimLoss()
    loss2 = Lloss(0.5)
    loss3 = TemporalLoss()

    x = [torch.randn((1, 1, 176, 176, 176)).cuda()]
    y = torch.randn((1, 1, 176, 176, 176)).cuda()
    loss1_val = loss1(x, y)
    loss2_val = loss2(x, y)
    loss3_val = loss3(x[0], y)
    print("ssim loss: ", loss1_val)
    print("l loss: ", loss2_val)
    print("temporal loss: ", loss3_val)







