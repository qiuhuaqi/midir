"""
Networks for FFD transformation model
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.base import conv_block_1, conv_blocks_2, conv_blocks_3
from model.networks.base import conv_Nd, avg_pool


class FFDNet(nn.Module):
    def __init__(self,
                 dim=2,
                 img_size=(192, 192),
                 cpt_spacing=(8, 8),
                 enc_channels=(64, 128, 256),
                 out_channels=(128, 64)
                 ):
        super(FFDNet, self).__init__()

        self.dim = dim

        # if one integer is given, assume same size for all dimensions
        if isinstance(img_size, int):
            self.img_size = (img_size,) * dim
        else:
            self.img_size = img_size

        if isinstance(cpt_spacing, int):
            self.cpt_spacing = (cpt_spacing,) * dim
        else:
            self.cpt_spacing = cpt_spacing

        # encoder layers
        self.enc = nn.ModuleList()
        self.enc.append(
            nn.Sequential(conv_Nd(self.dim, 2, enc_channels[0]),
                          nn.LeakyReLU(0.2),
                          avg_pool(self.dim),
                          )
        )
        for l in range(len(enc_channels) - 1):
            in_ch = enc_channels[l]
            out_ch = enc_channels[l + 1]
            self.enc.append(
                nn.Sequential(conv_Nd(self.dim, in_ch, out_ch),
                              nn.LeakyReLU(0.2),
                              avg_pool(self.dim))
            )

        # 1x1(x1) convolutions
        self.convs_out = nn.ModuleList()
        self.convs_out.append(
            nn.Sequential(conv_Nd(self.dim, enc_channels[-1], out_channels[0], kernel_size=1, padding=0),
                          nn.LeakyReLU(0.2))
        )
        for l in range(len(out_channels) - 1):
            in_ch = out_channels[l]
            out_ch = out_channels[l + 1]
            self.convs_out.append(
                nn.Sequential(conv_Nd(self.dim, in_ch, out_ch, kernel_size=1, padding=0),
                              nn.LeakyReLU(0.2))
            )

        # final output layer
        self.convs_out.append(
            conv_Nd(self.dim, out_channels[-1], self.dim, kernel_size=1, padding=0)
        )

    def _interpolate(self, x):
        # determine output size from image size and control point spacing
        self.output_size = tuple([int(isz // (cps + 1)) + 2
                             for isz, cps in zip(self.img_size, self.cpt_spacing)])
        inter_mode = "bilinear"
        if self.dim == 3:
            inter_mode = "bicubic"
        return F.interpolate(x, self.output_size, mode=inter_mode)

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        for enc in self.enc:
            x = enc(x)

        # re-sample to resize the feature map to output size
        x = self._interpolate(x)

        # 1x1(x1) conv and output
        for conv_out in self.convs_out:
            x = conv_out(x)
        return x


class SiameseFFDNet(nn.Module):
    def __init__(self,
                 dim=2,
                 ffd_cps=8,
                 enc_channels=(1, 64, 128, 256, 512, 512, 1024),
                 reduce_channel=512,
                 out_channels=(512 * 2, 512, 256, 64)
                 ):
        super(SiameseFFDNet, self).__init__()

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
        self.convs_out = nn.ModuleList()
        self.convs_out.append(
            nn.Conv2d(reduce_channel * (len(self.convs_reduce)), out_channels[0], 1))
        for l in range(len(out_channels) - 1):
            in_ch = out_channels[l]
            out_ch = out_channels[l + 1]
            self.convs_out.append(conv_blocks_2(in_ch, out_ch))

        # final output layer
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
            up_factor = 2 ** (l + 1)
            fm_upsampled.append(
                F.interpolate(fm_red, scale_factor=up_factor, mode="bilinear", align_corners=True))
        fm_concat = torch.cat(fm_upsampled, dim=1)

        # output conv layers
        output = fm_concat
        for conv_out in self.convs_out:
            output = conv_out(output)
        return output
