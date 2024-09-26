import torch
import torch.nn as nn


class Deconvolution(nn.Module):
    def __init__(self, in_channels=1, time_point=1000):
        super(Deconvolution, self).__init__()
        '''
        input: 
        video[b, 1, 1000, 24, 24]
        mask[b, 1, 24, 24]
        output:
        signal[b, 1, 1000]
        '''
        # main network
        self.block1 = TSA(in_channels, 8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block2 = TSA(8, 16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block3 = TSA(16, 32)

        # mask 分支网络
        self.mask_conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.mask_bn1 = nn.BatchNorm2d(8)
        self.mask_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mask_conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.mask_bn2 = nn.BatchNorm2d(16)
        self.mask_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mask_conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.mask_bn3 = nn.BatchNorm2d(32)

        self.avgpool = nn.AdaptiveAvgPool3d((time_point, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d((time_point, 1, 1))
        self.fc1 = nn.Linear(32*5*5, 32)

        # LSTM
        self.lstm = nn.LSTM(32, 4, batch_first=True)
        self.final = nn.Conv1d(4, 1, 5, 1, 2)

    def forward(self, x):
        video, mask = x[0], x[1]

        mask = nn.ReLU()(self.mask_bn1(self.mask_conv1(mask)))   # [b, 8, 24, 24]
        video = self.block1(video, mask)  # [b, 8, 1000, 24, 24]

        video = self.pool1(video)   # [b, 8, 1000, 12, 12]
        mask = self.mask_pool1(mask)   # [b, 8, 12, 12]

        mask = nn.ReLU()(self.mask_bn2(self.mask_conv2(mask)))   # [b, 16, 12, 12]
        video = self.block2(video, mask)   # [b, 16, 1000, 12, 12]

        video = self.pool2(video)   # [b, 16, 1000, 6, 6]
        mask = self.mask_pool2(mask)   # [b, 16, 6, 6]

        mask_p = nn.ReLU()(self.mask_bn3(self.mask_conv3(mask)))   # [b, 32, 6, 6]
        mask_n = nn.ReLU()(-self.mask_bn3(self.mask_conv3(mask)))
        video_p = self.block3(video, mask_p)   # [b, 32, 1000, 6, 6]
        video_n = self.block3(video, mask_n)
        video = video_p - video_n   # [b, 32, 1000, 6, 6]
        # video_temp = video.permute(0, 2, 1, 3, 4).contiguous().view(video.size(0), video.size(2), -1)  # [b, 1000, 32*5*5]
        # video_temp = self.fc1(video_temp)   # [b, 1000, 32]
        video = self.avgpool(video) + self.maxpool(video)   # [b, 32, 1000, 1, 1]
        signal = video.permute(0, 2, 1, 3, 4).squeeze(3).squeeze(3)  # [b, 1000, 32]
        signal, _ = self.lstm(signal)  # [b, 1000, 4]
        signal = self.final(signal.permute(0, 2, 1))  # [b, 1, 1000]
        return signal


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, hide_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hide_channels, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv3d(hide_channels, hide_channels, kernel_size=kernel_size,
                                   padding=padding, groups=hide_channels, bias=bias)
        self.pointwise = nn.Conv3d(hide_channels, out_channels, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(self.relu(shortcut + x))
        return x


class TSA(nn.Module):
    def __init__(self, in_planes, out_planes, bias=False):
        super(TSA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=bias, dilation=(1, 1, 1)),
            nn.Conv3d(out_planes, out_planes, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=bias, dilation=(1, 1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=(5, 1, 1), padding=(4, 0, 0), bias=bias, dilation=(2, 1, 1)),
            nn.Conv3d(out_planes, out_planes, kernel_size=(9, 1, 1), padding=(4, 0, 0), bias=bias, dilation=(1, 1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=(7, 1, 1), padding=(6, 0, 0), bias=bias, dilation=(2, 1, 1)),
            nn.Conv3d(out_planes, out_planes, kernel_size=(11, 1, 1), padding=(10, 0, 0), bias=bias, dilation=(2, 1, 1))
        )
        self.conv4 = nn.Conv3d(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(out_planes)
        self.dwconv = DepthwiseSeparableConv3d(out_planes, out_planes*2, out_planes)

    def forward(self, video_feature, mask_feature):
        encoder_feature = self.bn(self.relu(self.conv4(
            (self.conv1(video_feature) + self.conv2(video_feature) + self.conv3(video_feature)))))
        b, c, d, h, w = encoder_feature.size()
        prior_feature = mask_feature.unsqueeze(2).expand(-1, c, d, h, w)
        features = prior_feature * encoder_feature
        y = self.dwconv(features)
        return y


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = Deconvolution(in_channels=1)
    model.cuda()
    # summary(model, [(1, 1000, 20, 20), (1, 20, 20)])
    x = torch.randn((1, 1, 1000, 24, 24)).cuda()
    x_1 = torch.randn((1, 1, 24, 24)).cuda()
    y = model(x, x_1)
    print(y.shape)