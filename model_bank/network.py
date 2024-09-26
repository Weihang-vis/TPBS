from model_bank.unet_3d import UNet3D
from model_bank.nnet_3d import NNet3D, Uformer_NNet3D, Uformer_UNet3D
import torch.nn as nn


class Network_3D_Unet(nn.Module):
    def __init__(self, UNet_type='3DUNet', in_channels=1, out_channels=1, layer_order='cgr', f_maps=32, final_sigmoid=True):
        super(Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.layer_order = layer_order

        if UNet_type == '3DUNet':
            self.Generator = UNet3D(in_channels=in_channels, out_channels=out_channels, layer_order=layer_order,
                                    f_maps=f_maps, final_sigmoid=final_sigmoid)

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x


class Network_3D_Nnet(nn.Module):
    def __init__(self, UNet_type='3DNNet', in_channels=1, out_channels=1, f_maps=32, final_sigmoid=False):
        super(Network_3D_Nnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.f_maps = f_maps

        if UNet_type == '3DNNet':
            self.Generator = NNet3D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, final_sigmoid=final_sigmoid)

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x


class Network_Uformer_3D_Nnet(nn.Module):
    def __init__(self, UNet_type='Uformer_NNet3D', in_channels=1, out_channels=1, f_maps=64, final_sigmoid=False):
        super(Network_Uformer_3D_Nnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.f_maps = f_maps

        if UNet_type == 'Uformer_NNet3D':
            self.Generator = Uformer_NNet3D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, final_sigmoid=final_sigmoid)

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x


class Network_Uformer_3D_Unet(nn.Module):
    def __init__(self, UNet_type='Uformer_UNet3D', in_channels=1, out_channels=1, f_maps=32,
                 layer_order='cgr', final_sigmoid=False):
        super(Network_Uformer_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.f_maps = f_maps

        if UNet_type == 'Uformer_UNet3D':
            self.Generator = Uformer_UNet3D(in_channels=in_channels, out_channels=out_channels, layer_order=layer_order,
                                            f_maps=f_maps, final_sigmoid=final_sigmoid)

    def forward(self, x):
        fake_x = self.Generator(x)
        return fake_x