import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Networks
"""

class BaseNet(nn.Module):
    """
    The Network in paper:
    C. Qin et al., “Joint learning of motion estimation and segmentation for cardiac MR image sequences,” MICCAI 2018
    Network coding follows VoxelMorph preliminary Pytorch version:
        https://github.com/voxelmorph/voxelmorph/tree/8d57e6233da0141ca947f8c9d4c046f855bf904b/pytorch
    """

    def __init__(self,
                 dim=2,
                 enc_channels=(1, 64, 128, 256, 512, 512),
                 reduce_channel=64,
                 out_channels=(64, 64)
                 ):
        super(BaseNet, self).__init__()

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
        self.conv_reduce = nn.ModuleList()
        for ch in enc_channels[1:]:
            self.conv_reduce.append(conv_block_1(ch * 2, reduce_channel))

        # final convs on concatenated feature maps
        self.conv_out = nn.ModuleList()
        self.conv_out.append(
            nn.Conv2d(reduce_channel * (len(enc_channels) - 1), out_channels[0], 1))
        self.conv_out.append(conv_block_1(out_channels[0], out_channels[1]))
        self.conv_out.append(nn.Conv2d(out_channels[-1], dim, 1))

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

        # concat feature maps from two streams and reduce to the same channel
        fm_reduce = list()
        for l, (fm_t, fm_s) in enumerate(zip(fm_tar[1:], fm_src[1:])):
            fm_reduce.append(
                self.conv_reduce[l](torch.cat((fm_t, fm_s), 1)))

        # upsample all to full resolution and concatenate
        up_factors = [2 ** (i + 1) for i in range(len(self.enc_tar) - 1)]
        fm_upsampled = [fm_reduce[0]]
        for l, fm_red in enumerate(fm_reduce[1:]):
            fm_upsampled.append(F.interpolate(fm_red, scale_factor=up_factors[l], mode="bilinear", align_corners=True))
        fm_concat = torch.cat(fm_upsampled, 1)

        # output conv layers
        output = fm_concat
        for conv_out in self.conv_out:
            output = conv_out(output)
        return output


class BaseNetFFD(nn.Module):
    """Deformable registration network with input from image space
    The Network in paper:
    C. Qin et al., “Joint learning of motion estimation and segmentation for cardiac MR image sequences,” MICCAI 2018
    Modified for FFD model (to keep network capacity while output resolution is lower)
    """

    def __init__(self):
        super().__init__()
        self.conv_blocks1 = conv_blocks_2(1, 64)
        self.conv_blocks2 = conv_blocks_2(64, 128, 2)
        self.conv_blocks3 = conv_blocks_3(128, 256, 2)
        self.conv_blocks4 = conv_blocks_3(256, 512, 2)
        self.conv_blocks5 = conv_blocks_3(512, 512, 2)
        self.conv_blocks6 = conv_blocks_3(512, 1024, 2)  # 1/32

        self.conv_blocks12 = conv_blocks_2(1, 64)
        self.conv_blocks22 = conv_blocks_2(64, 128, 2)
        self.conv_blocks32 = conv_blocks_3(128, 256, 2)
        self.conv_blocks42 = conv_blocks_3(256, 512, 2)
        self.conv_blocks52 = conv_blocks_3(512, 512, 2)

        self.conv_blocks62 = conv_blocks_3(512, 1024, 2)  # 1/32

        # self.conv1 = conv_block_1(128, 64)
        # self.conv2 = conv_block_1(256, 64)
        # self.conv3 = conv_block_1(512, 64)
        self.conv4 = conv_block_1(1024, 512)
        self.conv5 = conv_block_1(1024, 512)
        self.conv6 = conv_block_1(2048, 512)

        self.conv7 = nn.Conv2d(512 * 3, 512 * 2, 1)
        self.conv8 = conv_blocks_2(512 * 2, 512)
        self.conv9 = conv_blocks_2(512, 256)
        self.conv10 = conv_blocks_2(256, 64)
        self.conv_output = nn.Conv2d(64, 2, 1)

    def forward(self, target, source):
        """
        Forward pass function.

        Args:
            target: target image input to the network, Tensor shape (T-1, 1, H, W)
            source: source image input to the network, Tensor shape (T-1, 1, H, W)

        Returns:
            net['out']: (Tensor, shape (N, 2, H, W)) output of the network
        """
        net = {}

        # two branches, weights not shared
        net['conv1s'] = self.conv_blocks1(source)
        net['conv2s'] = self.conv_blocks2(net['conv1s'])
        net['conv3s'] = self.conv_blocks3(net['conv2s'])
        net['conv4s'] = self.conv_blocks4(net['conv3s'])
        net['conv5s'] = self.conv_blocks5(net['conv4s'])
        net['conv6s'] = self.conv_blocks6(net['conv5s'])

        net['conv1t'] = self.conv_blocks12(target)
        net['conv2t'] = self.conv_blocks22(net['conv1t'])
        net['conv3t'] = self.conv_blocks32(net['conv2t'])
        net['conv4t'] = self.conv_blocks42(net['conv3t'])
        net['conv5t'] = self.conv_blocks52(net['conv4t'])
        net['conv6t'] = self.conv_blocks62(net['conv5t'])

        # concatenate feature maps of the two branches
        # net['concat1'] = torch.cat((net['conv1s'], net['conv1t']), 1)
        # net['concat2'] = torch.cat((net['conv2s'], net['conv2t']), 1)
        # net['concat3'] = torch.cat((net['conv3s'], net['conv3t']), 1)
        net['concat4'] = torch.cat((net['conv4s'], net['conv4t']), 1)
        net['concat5'] = torch.cat((net['conv5s'], net['conv5t']), 1)
        net['concat6'] = torch.cat((net['conv6s'], net['conv6t']), 1)

        # convolution on the concatenated feature maps
        # net['out1'] = self.conv1(net['concat1'])
        # net['out2'] = self.conv2(net['concat2'])
        # net['out3'] = self.conv3(net['concat3'])
        net['out4'] = self.conv4(net['concat4'])  # 1/8
        net['out5'] = self.conv5(net['concat5'])  # 1/16
        net['out6'] = self.conv6(net['concat6'])  # 1/32

        # upsample to image size / cps x down sample for FFD
        net['out5_up'] = F.interpolate(net['out5'], scale_factor=2, mode='bilinear', align_corners=True)  # 1/8
        net['out6_up'] = F.interpolate(net['out6'], scale_factor=4, mode='bilinear', align_corners=True)  # 1/8
        net['concat'] = torch.cat((net['out4'], net['out5_up'], net['out6_up']), 1)

        # final convolution and output
        net['comb_1'] = self.conv7(net['concat'])
        net['comb_2'] = self.conv8(net['comb_1'])
        net['comb_3'] = self.conv9(net['comb_2'])
        net['comb_4'] = self.conv10(net['comb_3'])
        net['out'] = self.conv_output(net['comb_4'])

        return net['out']


""""""

"""
Sub-network modules
"""


def relu():
    return nn.ReLU(inplace=True)


def conv_block_1(in_channels, out_channels, kernel_size=3, stride=1, padding=1, nonlinearity=relu):
    """Conv2d + Non-linearity + BN2d, Xavier initialisation"""
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    nn.init.xavier_uniform_(conv_layer.weight, gain=np.sqrt(2.0))

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2(in_channels, out_channels, strides=1):
    """Block of 2x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride=strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3(in_channels, out_channels, strides=1):
    """Block of 3x Conv layers"""
    conv1 = conv_block_1(in_channels, out_channels, stride=strides)
    conv2 = conv_block_1(out_channels, out_channels, stride=1)
    conv3 = conv_block_1(out_channels, out_channels, stride=1)
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)
""""""
