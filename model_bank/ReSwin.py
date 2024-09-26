import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

import logging


# 可用InputProj替代
class PatchEmbed(nn.Module):
    """ Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class InputProj(nn.Module):
    def __init__(self, in_channel=1, out_channel=32, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            H, W = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.out_channel, H, W)
        return x


# 可用LeFF改进
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):

        y = self.avg_pool(rearrange(x, ' b h w c -> b c h w '))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer(dim) if use_eca else nn.Identity()

    def forward(self, x):
        """
        Args:
             x: Input feature, tensor size (B, H, W, C).
        Returns:
             x: Output feature, tensor size (B, H, W, C).
        """
        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b h w c -> b c h w ')
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b h w c')
        x = self.linear2(x)
        x = self.eca(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size[1]*window_size[2], C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0],
               W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    x = windows.view(B, H // window_size[0], W // window_size[1],
                     window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].\
            reshape(N, N, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            # softmax seems like not good
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            pass

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False, modulator=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        if modulator:
            self.modulator = nn.Embedding(window_size[0] * window_size[1], dim)
        else:
            self.modulator = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = LeFF(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    # attention

    def forward_part1(self, x, mask_matrix):
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wh*Ww, C
        # with_modulator
        if self.modulator is not None:
            wmx_windows = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmx_windows = x_windows
        # W-MSA/SW-MSA
        attn_windows = self.attn(wmx_windows, mask=attn_mask)  # B*nW, Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    # Mlp
    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


# cache each stage results
@lru_cache()
def compute_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 Hp Wp 1
    cnt = 0
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


# downsample 可用卷积替代
class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Downsample(nn.Module):
    def __init__(self, dim, norm_layer=None):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# upsample 可用卷积替代
class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, dim_scale*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w-> b h w c')
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//2)
        x = self.norm(x)
        x = rearrange(x, 'b d h w c-> b c d h w')

        return x


class Upsample(nn.Module):
    def __init__(self, dim, norm_layer=None):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, depth, num_heads, window_size=(7, 7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, H, W).
        """
        # calculate attention mask for SW-MSA
        short_cut = x
        B, C, H, W = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c h w -> b h w c')
        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask = compute_mask(Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, H, W, -1)
        x = rearrange(x, 'b h w c -> b c h w')
        x = x + short_cut
        return x


# 可用卷积替代
class FinalPatchExpand(nn.Module):
    def __init__(self, dim, patch_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.dim_scale = reduce((lambda x, y: x * y), patch_size)
        self.norm = norm_layer(dim//self.dim_scale)

    def forward(self, x):

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w-> b h w c')

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.patch_size[0], p2=self.patch_size[1], c=C//self.dim_scale)
        # x = self.norm(x)
        x = rearrange(x, 'b h w c-> b c h w')

        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel=32, out_channel=1, kernel_size=3, stride=1, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))

    def forward(self, x):
        x = self.proj(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = self.ChannelAttention(in_planes, ratio)
        self.sa = self.SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio):
            super(CBAM.ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
            )

        def forward(self, x):
            avg_out = self.shared_MLP(self.avg_pool(x))
            max_out = self.shared_MLP(self.max_pool(x))
            out = avg_out + max_out
            return torch.sigmoid(out)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size):
            super(CBAM.SpatialAttention, self).__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            concat_out = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(concat_out)
            return torch.sigmoid(out)


class PriorSwinTransformer(nn.Module):

    def __init__(self, patch_size=(4, 4), in_chans=1, embed_dim=64,
                 depths=[2, 4, 4], num_heads=[2, 4, 8],
                 window_size=(8, 8), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False, frozen_stages=-1, use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            downsample = Downsample(dim=int(embed_dim * 2**i_layer))
            self.layers.append(nn.Sequential(layer, downsample))

        self.num_features = int(embed_dim * 2 ** (self.num_layers))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        # spatial attention pyramid
        self.pyramids = nn.ModuleList(
            CBAM(in_planes=int(embed_dim * 2**i_layer)) for i_layer in range(self.num_layers + 1)
        )

    def forward(self, x):
        # x = self.patch_embed(x)
        x = self.input_proj(x)
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x.contiguous())

        x = rearrange(x, 'n c h w -> n h w c')
        x = self.norm(x)
        x = rearrange(x, 'n h w c -> n c h w')
        x_downsample.append(x)
        # spatial pyramid
        spatial_pyramids = [pyramid(x) for pyramid, x in zip(self.pyramids, x_downsample)]

        return spatial_pyramids


class Uformer(nn.Module):

    def __init__(self, patch_size=(4, 4), in_chans=1, embed_dim=64,
                 depths=[2, 4, 4], num_heads=[2, 4, 8],
                 window_size=(8, 8), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False, frozen_stages=-1, use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            downsample = Downsample(dim=int(embed_dim * 2**i_layer))
            self.layers.append(nn.Sequential(layer, downsample))

        self.num_features = int(embed_dim * 2 ** (self.num_layers))

        # Bottleneck
        self.bottleneck = BasicLayer(
            dim=int(embed_dim * 2 ** self.num_layers), depth=depths[self.num_layers - 1],
            num_heads=num_heads[self.num_layers - 1],
            window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        # build decoder layers
        self.layers_up = nn.ModuleList()

        for i_layer in range(self.num_layers):
            upsample = Upsample(dim=int(embed_dim * 2 ** (self.num_layers - i_layer)))
            conv = nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - i_layer)),
                             int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), 3, 1, 1)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                depth=depths[(self.num_layers - 1 - i_layer)],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                    depths[:(self.num_layers - 1 - i_layer) + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            self.layers_up.append(nn.ModuleList([upsample, conv, layer]))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # self.up = FinalPatchExpand3D(dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        self.pyramids = nn.ModuleList(
            CBAM(in_planes=int(embed_dim * 2**i_layer)) for i_layer in range(self.num_layers + 1)
        )

    def forward_features(self, x):
        # x = self.patch_embed(x)
        x = self.input_proj(x)
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x.contiguous())

        x = rearrange(x, 'n c h w -> n h w c')
        x = self.norm(x)
        x = rearrange(x, 'n h w c -> n c h w')
        x_downsample.append(x)

        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):

            upsample, conv, layer = layer_up
            x = upsample(x)
            x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], 1)
            x = conv(x)
            x = layer(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm_up(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x

    def forward(self, x):
        y, x_downsample = self.forward_features(x)
        # spatial pyramid
        spatial_pyramids = [pyramid(x) for pyramid, x in zip(self.pyramids, x_downsample)]
        y = self.bottleneck(y)
        y = self.forward_up_features(y, x_downsample)
        y = self.output_proj(y)

        return y, spatial_pyramids


class Uformer_pyramid(nn.Module):

    def __init__(self, patch_size=(4, 4), in_chans=1, embed_dim=64,
                 depths=[2, 4, 4], num_heads=[2, 4, 8],
                 window_size=(8, 8), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False, frozen_stages=-1, use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            downsample = Downsample(dim=int(embed_dim * 2**i_layer))
            self.layers.append(nn.Sequential(layer, downsample))

        self.num_features = int(embed_dim * 2 ** (self.num_layers))

        # Bottleneck
        self.bottleneck = BasicLayer(
            dim=int(embed_dim * 2 ** self.num_layers), depth=depths[self.num_layers - 1],
            num_heads=num_heads[self.num_layers - 1],
            window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        # build decoder layers
        self.layers_up = nn.ModuleList()

        for i_layer in range(self.num_layers):
            upsample = Upsample(dim=int(embed_dim * 2 ** (self.num_layers - i_layer)))
            conv = nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - i_layer)),
                             int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), 3, 1, 1)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                depth=depths[(self.num_layers - 1 - i_layer)],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                    depths[:(self.num_layers - 1 - i_layer) + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            self.layers_up.append(nn.ModuleList([upsample, conv, layer]))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # self.up = FinalPatchExpand3D(dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=1, kernel_size=3, stride=1)

        self.pyramids_down = nn.ModuleList(
            CBAM(in_planes=int(embed_dim * 2 ** i_layer)) for i_layer in range(self.num_layers + 1)
        )
        self.pyramids_up = nn.ModuleList(
            CBAM(in_planes=int(embed_dim * 2 ** (self.num_layers - i_layer))) for i_layer in range(self.num_layers + 1)
        )

    def forward_features(self, x):
        # x = self.patch_embed(x)
        x = self.input_proj(x)
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x.contiguous())

        x = rearrange(x, 'n c h w -> n h w c')
        x = self.norm(x)
        x = rearrange(x, 'n h w c -> n c h w')
        x_downsample.append(x)

        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        x_upsample = []
        for inx, layer_up in enumerate(self.layers_up):
            x_upsample.append(x)

            upsample, conv, layer = layer_up
            x = upsample(x)
            x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], 1)
            x = conv(x)
            x = layer(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm_up(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x_upsample.append(x)

        return x, x_upsample

    def forward(self, x):
        y, x_downsample = self.forward_features(x)
        # spatial pyramid
        spatial_pyramids_down = [pyramid(x) for pyramid, x in zip(self.pyramids_down, x_downsample)]
        y = self.bottleneck(y)
        y, x_upsample = self.forward_up_features(y, x_downsample)
        spatial_pyramids_up = [pyramid(x) for pyramid, x in zip(self.pyramids_up, x_upsample)]
        y = self.output_proj(y)

        return y, [spatial_pyramids_down, spatial_pyramids_up]


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    model = Uformer_pyramid(in_chans=1, embed_dim=64)
    model.cuda()
    summary(model, (1, 176, 176))
    x = torch.randn((1, 1, 176, 176)).cuda()
    y, [spatial_pyramids_down, spatial_pyramids_up] = model(x)
    print(y.shape)







