import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from model.networks.base import conv_blocks_2, conv_blocks_3, conv_block_1

"""
Networks for FFD transformation model
"""

class FFDNet(nn.Module):
    def __init__(self):
        super(FFDNet, self).__init__()

    def forward(self, tar, src):
        pass


class NewSiameseFFDNet(nn.Module):
    def __init__(self,
                 dim=2,
                 ffd_cps=8,
                 enc_channels=(1, 64, 128, 256, 512, 512, 1024),
                 reduce_channel=512,
                 out_channels=(512*2, 512, 256, 64)
                 ):
        super(NewSiameseFFDNet, self).__init__()

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

        # final conv layers on concatenated feature maps
        self.convs_out = nn.ModuleList()
        self.convs_out.append(
            nn.Conv2d(reduce_channel * (len(self.convs_reduce)), out_channels[0], 1))
        for l in range(len(out_channels) - 1):
            in_ch = out_channels[l]
            out_ch = out_channels[l + 1]
            self.convs_out.append(conv_blocks_2(in_ch, out_ch))

        self.convs_out.append(nn.Conv2d(out_channels[-1], dim, 1))

    def forward(self, target, source):
        """
        Forward pass function.

        Args:
            target: target image input to the network, Tensor shape (T-1, 1, H, W)
            source: source image input to the network, Tensor shape (T-1, 1, H, W)

        Returns:
            net['out']: (Tensor, shape (N, 2, H, W)) output of the network
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
        for l, (fm_t, fm_s) in enumerate(zip(fm_tar[self.start_reduce_layer:], fm_src[self.start_reduce_layer:])):
            fm_reduce.append(
                self.convs_reduce[l](torch.cat((fm_t, fm_s), 1)))

        # upsample all to full resolution and concatenate
        fm_upsampled = [fm_reduce[0]]
        for l, fm_red in enumerate(fm_reduce[1:]):
            up_factor = 2 ** (l+1)
            fm_upsampled.append(
                F.interpolate(fm_red, scale_factor=up_factor, mode="bilinear", align_corners=True))
        fm_concat = torch.cat(fm_upsampled, 1)

        # output conv layers
        output = fm_concat
        for conv_out in self.convs_out:
            output = conv_out(output)
        return output


class SiameseFFDNet(nn.Module):
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

    def forward(self, tar, src):
        """
        Forward pass function.

        Args:
            tar: target image input to the network, Tensor shape (T-1, 1, H, W)
            src: source image input to the network, Tensor shape (T-1, 1, H, W)

        Returns:
            net['out']: (Tensor, shape (N, 2, H, W)) output of the network
        """
        net = {}

        # two branches, weights not shared
        net['conv1s'] = self.conv_blocks1(src)
        net['conv2s'] = self.conv_blocks2(net['conv1s'])
        net['conv3s'] = self.conv_blocks3(net['conv2s'])
        net['conv4s'] = self.conv_blocks4(net['conv3s'])
        net['conv5s'] = self.conv_blocks5(net['conv4s'])
        net['conv6s'] = self.conv_blocks6(net['conv5s'])

        net['conv1t'] = self.conv_blocks12(tar)
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