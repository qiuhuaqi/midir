import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# ---
# Legacy Siamese networks, some not adapted to 3D
# ---


class SiameseNet(nn.Module):
    """
    The Network in paper:
    C. Qin et al., “Joint learning of motion estimation and segmentation for cardiac MR image sequences,” MICCAI 2018
    2D only
    """

    def __init__(self,
                 dim=2,
                 enc_channels=(1, 64, 128, 256, 512, 512),
                 # todo: in later code the first encoder channel is not included,
                 #  since it's always known 1 or 2
                 reduce_channel=64,
                 out_channels=(64, 64)
                 ):
        super(SiameseNet, self).__init__()

        # dual-stream encoders
        self.enc_tar = nn.ModuleList()
        self.enc_src = nn.ModuleList()

        self.enc_tar.append(conv_blocks_2(enc_channels[0], enc_channels[1]))
        self.enc_src.append(conv_blocks_2(enc_channels[0], enc_channels[1]))
        for l in range(len(enc_channels) - 2):
            in_ch = enc_channels[l + 1]
            out_ch = enc_channels[l + 2]
            self.enc_tar.append(conv_blocks_3(in_ch, out_ch, 2))
            self.enc_src.append(conv_blocks_3(in_ch, out_ch, 2))

        # conv layers to reduce to the same number of channels
        self.convs_reduce = nn.ModuleList()
        for ch in enc_channels[1:]:
            self.convs_reduce.append(conv_block_1(ch * 2, reduce_channel))

        # final convs on concatenated feature maps
        self.convs_out = nn.ModuleList()
        self.convs_out.append(nn.Conv2d(reduce_channel * (len(enc_channels) - 1), out_channels[0], 1))
        self.convs_out.append(conv_block_1(out_channels[0], out_channels[1]))
        self.convs_out.append(nn.Conv2d(out_channels[-1], dim, 1))

    def forward(self, target, source):
        """
        Args:
            target: (torch.Tensor, shape (N, 1, H, W)) target input image
            source: (torch.Tensor, shape (N, 1, H, W)) source input image
        Returns:
            out: (torch.Tensor, shape (N, 1, *) network output
        """
        # compute encoder feature maps
        fm_tar = [target]
        fm_src = [source]
        for (enc_t, enc_s) in zip(self.enc_tar, self.enc_src):
            fm_tar.append(enc_t(fm_tar[-1]))
            fm_src.append(enc_s(fm_src[-1]))

        # concat feature maps from two streams
        # and reduce to the same channel via a conv layer
        fm_reduce = list()
        for l, (fm_t, fm_s) in enumerate(zip(fm_tar[1:], fm_src[1:])):
            fm_reduce.append(
                self.convs_reduce[l](torch.cat((fm_t, fm_s), 1)))

        # upsample all to full resolution and concatenate
        fm_upsampled = [fm_reduce[0]]
        for l, fm_red in enumerate(fm_reduce[1:]):
            up_factor = 2 ** (l + 1)
            fm_upsampled.append(
                F.interpolate(fm_red, scale_factor=up_factor, mode="bilinear", align_corners=False))
        fm_concat = torch.cat(fm_upsampled, 1)

        # output conv layers
        output = fm_concat
        for conv_out in self.convs_out:
            output = conv_out(output)
        return output


class SiameseNetFFD(nn.Module):
    """Modification of the Siamese network,
    3D enabled code but memory consuming.
    """
    def __init__(self,
                 dim=2,
                 ffd_cps=8,
                 enc_channels=(1, 64, 128, 256, 512, 512, 1024),
                 reduce_channel=512,
                 out_channels=(512 * 2, 512, 256, 64)
                 ):
        super(SiameseNetFFD, self).__init__()

        # dual-stream encoders
        self.enc_tar = nn.ModuleList()
        self.enc_src = nn.ModuleList()

        self.enc_tar.append(conv_blocks_2(enc_channels[0], enc_channels[1]))
        self.enc_src.append(conv_blocks_2(enc_channels[0], enc_channels[1]))
        for l in range(len(enc_channels) - 2):
            in_ch = enc_channels[l + 1]
            out_ch = enc_channels[l + 2]
            self.enc_tar.append(conv_blocks_3(in_ch, out_ch, 2))
            self.enc_src.append(conv_blocks_3(in_ch, out_ch, 2))

        # determine which layers to use for FFD using the control point spacing setting
        assert math.log2(ffd_cps) % 1 == 0, "FFD control point spacing is not an power of 2"
        # 4 == log2(cps) + 1 (first layer no downsample)
        self.start_reduce_layer = int(math.log2(ffd_cps)) + 1  # first layer not downsampling)

        # conv layers to reduce to the same number of channels
        self.convs_reduce = nn.ModuleList()
        for ch in enc_channels[self.start_reduce_layer:]:
            self.convs_reduce.append(conv_block_1(ch * 2, reduce_channel))

        # conv layers on concatenated feature maps
        self.out_layers = nn.ModuleList()
        self.out_layers.append(
            nn.Conv2d(reduce_channel * (len(self.convs_reduce)), out_channels[0], 1))
        for l in range(len(out_channels) - 1):
            in_ch = out_channels[l]
            out_ch = out_channels[l + 1]
            self.out_layers.append(conv_blocks_2(in_ch, out_ch))

        # final output layer
        self.out_layers.append(nn.Conv2d(out_channels[-1], dim, 1))

    def forward(self, tar, src):
        """
        Forward pass function.

        Args:
            tar: (Tensor shape (N, 1, H, W)) target image input to the network
            src: (Tensor shape (N, 1, H, W)) source image input to the network
        Returns:
            y: (Tensor shape (N, 2, cH, cW)) control point parameters
        """
        # compute encoder feature maps
        fm_tar = [tar]
        fm_src = [src]
        for (enc_t, enc_s) in zip(self.enc_tar, self.enc_src):
            fm_tar.append(enc_t(fm_tar[-1]))
            fm_src.append(enc_s(fm_src[-1]))

        # concat feature maps from two streams
        # and reduce to the same channel via a conv layer
        fm_reduce = list()
        for l, (fm_t, fm_s) in enumerate(zip(fm_tar[self.start_reduce_layer:], fm_src[self.start_reduce_layer:])):
            fm_reduce.append(
                self.convs_reduce[l](torch.cat((fm_t, fm_s), 1)))

        # upsample all to full resolution and concatenate
        fm_upsampled = [fm_reduce[0]]
        for l, fm_red in enumerate(fm_reduce[1:]):
            up_factor = 2 ** (l + 1)
            fm_upsampled.append(
                F.interpolate(fm_red, scale_factor=up_factor, mode="bilinear", align_corners=True))
        fm_concat = torch.cat(fm_upsampled, dim=1)

        # output conv layers
        y = fm_concat
        for out_layer in self.out_layers:
            y = out_layer(y)
        return y


def conv_block_1(in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
    """Conv2d + Non-linearity + BN2d, Xavier initialisation"""
    conv_layer = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=False)
    nn.init.xavier_uniform_(conv_layer.weight, gain=np.sqrt(2.0))

    bn_layer = nn.BatchNorm2d(out_channels)
    nll_layer = nn.ReLU(inplace=True)
    return nn.Sequential(conv_layer, bn_layer, nll_layer)


def conv_blocks_2(in_channels,
                  out_channels,
                  strides=1):
    """Block of 2x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride=strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    return nn.Sequential(conv1, conv2)


def conv_blocks_3(in_channels,
                  out_channels,
                  strides=1):
    """Block of 3x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride=strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    conv3 = conv_block_1(out_channels, out_channels, stride=1)
    return nn.Sequential(conv1, conv2, conv3)