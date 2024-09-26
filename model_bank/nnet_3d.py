import torch
from torch import nn as nn
from torch.nn import functional as F
from utils import create_feature_maps
from model_bank.ReSwin import PriorSwinTransformer, Uformer, Uformer_pyramid


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder: bool, kernel_size=3, order='cr', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='cr',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels, encoder=True,
                                         kernel_size=conv_kernel_size, order=conv_layer_order, num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                               stride=scale_factor, padding=1, output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels, encoder=False, kernel_size=kernel_size,
                                         order=conv_layer_order, num_groups=num_groups)

    def forward(self, encoder_features, pyramid_feature, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            # x = torch.cat((encoder_features + pyramid_feature, x), dim=1)   # for Nnet
            x = torch.cat((encoder_features, pyramid_feature + x), dim=1)  # for Uformer_nnet_3d
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)
        return x


class TSA(nn.Module):
    def __init__(self, in_planes):
        super(TSA, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        # NNet does not use this
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, encoder_feature, prior_feature):
        _, c, d, h, w = encoder_feature.size()
        encoder_feature = self.conv2(self.conv1(encoder_feature))
        # encoder_feature = self.conv1(encoder_feature)
        prior_feature = prior_feature.unsqueeze(2).expand(-1, c, d, h, w)
        return prior_feature * encoder_feature


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, hide_channels, out_channels, kernel_size=3, padding=1, group_num=8, bias=False):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hide_channels, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv3d(hide_channels, hide_channels, kernel_size=kernel_size,
                                   padding=padding, groups=hide_channels, bias=bias)
        self.pointwise = nn.Conv3d(hide_channels, out_channels, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.gn = nn.GroupNorm(group_num, out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gn(self.relu(shortcut + x))
        return x


class TSA_plus(nn.Module):
    def __init__(self, in_planes, group_num=8, bias=False):
        super(TSA_plus, self).__init__()
        # self.TFEncoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=in_planes, nhead=4, dim_feedforward=2*in_planes), num_layers=1)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=bias, dilation=(1, 1, 1)),
            nn.Conv3d(in_planes, in_planes, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=bias, dilation=(1, 1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size=(5, 1, 1), padding=(4, 0, 0), bias=bias, dilation=(2, 1, 1)),
            nn.Conv3d(in_planes, in_planes, kernel_size=(9, 1, 1), padding=(4, 0, 0), bias=bias, dilation=(1, 1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size=(7, 1, 1), padding=(6, 0, 0), bias=bias, dilation=(2, 1, 1)),
            nn.Conv3d(in_planes, in_planes, kernel_size=(11, 1, 1), padding=(10, 0, 0), bias=bias, dilation=(2, 1, 1))
        )
        self.conv4 = nn.Conv3d(in_planes, in_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.gn = nn.GroupNorm(group_num, in_planes)
        self.dwconv = DepthwiseSeparableConv3d(in_planes, in_planes*2, in_planes, group_num=group_num)

    def forward(self, encoder_feature, prior_feature):
        # TODO: do more temporal operations
        b, c, d, h, w = encoder_feature.size()
        # encoder_feature = encoder_feature.permute(2, 0, 3, 4, 1).contiguous().view(d, b*h*w, c)
        # encoder_feature = self.TFEncoder(encoder_feature)
        # encoder_feature = encoder_feature.view(d, b, h, w, c).permute(1, 4, 0, 2, 3).contiguous()

        encoder_feature = self.gn(self.relu(self.conv4(
            (self.conv1(encoder_feature) + self.conv2(encoder_feature) + self.conv3(encoder_feature)))))
        prior_feature = prior_feature.unsqueeze(2).expand(-1, c, d, h, w)
        features = prior_feature * encoder_feature
        y = self.dwconv(features)
        return y


class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr', num_groups=8):
        super(FinalConv, self).__init__()

        # conv1
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        # in the last layer a 1×1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)


class NNet3D(nn.Module):

    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8, **kwargs):
        super(NNet3D, self).__init__()
        self.embed_dim = f_maps

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.pyramids = PriorSwinTransformer(in_chans=1, embed_dim=self.embed_dim)
        self.TSA = nn.ModuleList([TSA(m) for m in f_maps])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x_mean = torch.mean(x, 2)
        spatial_pyramids = self.pyramids(x_mean)
        # encoder part
        encoders_features = []
        spatial_pyramids_new = []
        for encoder, TSA in zip(self.encoders, self.TSA):
            x = encoder(x)
            spatial_pyramids_new.insert(0, TSA(x, spatial_pyramids.pop(0)))
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        x = x + spatial_pyramids_new[0]
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        spatial_pyramids_new = spatial_pyramids_new[1:]

        # decoder part
        for decoder, encoder_features, spatial_pyramid in zip(self.decoders, encoders_features, spatial_pyramids_new):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, spatial_pyramid, x)

        x = self.final_conv(x)

        return [x]


class Uformer_NNet3D(nn.Module):

    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8, **kwargs):
        super(Uformer_NNet3D, self).__init__()
        self.embed_dim = f_maps

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.pyramids = Uformer(in_chans=1, embed_dim=self.embed_dim)
        self.TSA = nn.ModuleList([TSA(m) for m in f_maps])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x_mean = torch.mean(x, 2)
        y_mean, spatial_pyramids = self.pyramids(x_mean)
        # encoder part
        encoders_features = []
        spatial_pyramids_new = []
        for encoder, TSA in zip(self.encoders, self.TSA):
            x = encoder(x)
            spatial_pyramids_new.insert(0, TSA(x, spatial_pyramids.pop(0)))
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        x = x + spatial_pyramids_new[0]
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        spatial_pyramids_new = spatial_pyramids_new[1:]

        # decoder part
        for decoder, encoder_features, spatial_pyramid in zip(self.decoders, encoders_features, spatial_pyramids_new):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, spatial_pyramid, x)

        x = self.final_conv(x)

        return [x, y_mean]


class Uformer_UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='crg', num_groups=8, **kwargs):
        super(Uformer_UNet3D, self).__init__()
        self.embed_dim = f_maps

        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(f_maps[-1], 2 * f_maps[-1], kernel_size=1),
            nn.Conv3d(2 * f_maps[-1], 2 * f_maps[-1], kernel_size=3, padding=1, groups=2 * f_maps[-1]),
            nn.Conv3d(2 * f_maps[-1], f_maps[-1], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=f_maps[-1])
        )

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        self.pyramids = Uformer_pyramid(in_chans=in_channels, embed_dim=self.embed_dim)
        self.TSA_down = nn.ModuleList([TSA_plus(m) for m in f_maps])
        self.TSA_up = nn.ModuleList([TSA_plus(m) for m in f_maps[::-1]])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x_mean = torch.mean(x, 2)
        y_mean, [spatial_pyramids_down, spatial_pyramids_up] = self.pyramids(x_mean)
        # encoder part
        encoders_features = []
        for encoder, TSA, spatial_pyramid_down in zip(self.encoders, self.TSA_down, spatial_pyramids_down):
            x = encoder(x)
            x = x + TSA(x, spatial_pyramid_down)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # bottleneck
        x = self.bottleneck(x)
        # decoder part
        for decoder, encoder_feature, spatial_pyramid_up, TSA in zip(self.decoders, encoders_features,
                                                                     spatial_pyramids_up, self.TSA_up):
            x = decoder(encoder_feature,
                        F.interpolate(TSA(x, spatial_pyramid_up), size=encoder_feature.size()[2:], mode='nearest'), x)

        x = self.final_conv(x)

        return [x, y_mean]


class ResidualUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
        skip_final_activation (bool): if True, skips the final normalization layer (sigmoid/softmax) and returns the
            logits directly
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, conv_layer_order='cge', num_groups=8,
                 skip_final_activation=False, **kwargs):
        super(ResidualUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses ExtResNetBlock as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses ExtResNetBlock as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if not skip_final_activation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = Uformer_UNet3D(in_channels=1, out_channels=1, fmaps=32, final_sigmoid=False)
    model.cuda()
    # summary(model, (1, 176, 176, 176))
    x = torch.randn((1, 1, 176, 176, 176)).cuda()
    [y, _] = model(x)
    print(y.shape)