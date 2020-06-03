""" Networks for DVF transformation model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.base import conv_Nd
from model.networks.base import conv_block_1, conv_blocks_2, conv_blocks_3


""" 2D/3D """

class UNet(nn.Module):
    """
    Slight modification of the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 dim=2,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32, 16),
                 out_channels=(16, 16)
                 ):
        super(UNet, self).__init__()

        self.dim = dim

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            self.enc.append(
                nn.Sequential(
                    conv_Nd(dim, in_ch, enc_channels[i], stride=2),
                    nn.LeakyReLU(0.2)
                )
            )

        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i-1] + enc_channels[-2-(i-1)]
            self.dec.append(
                nn.Sequential(
                    conv_Nd(dim, in_ch, dec_channels[i]),  # stride=1
                    nn.LeakyReLU(0.2)
                )
            )

        # upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # conv layers before prediction
        self.out_layers = nn.ModuleList()
        for i in range(len(out_channels)):
            in_ch = dec_channels[-1] + enc_channels[0] if i == 0 else out_channels[i-1]
            self.out_layers.append(
                nn.Sequential(
                    conv_Nd(dim, in_ch, out_channels[i]),  # stride=1
                    nn.LeakyReLU(0.2)
                )
            )

        # final prediction layer
        self.out_layers.append(
            conv_Nd(out_channels[-1], dim)
        )


    def forward(self, tar, src):
        # concat
        x = torch.cat((tar, src), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate series (to full resolution)
        fm_dec = fm_enc[-1]
        for i, dec in enumerate(self.dec):
            fm_dec = dec(fm_dec)
            fm_dec = self.upsample(fm_dec)
            fm_dec = torch.cat([fm_dec, fm_enc[-2-i]], dim=1)

        # further convs and prediction
        y = fm_dec[-1]
        for out_layer in self.out_layers:
            y = out_layer(y)
        return y



""" 2D only """

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
                F.interpolate(fm_red, scale_factor=up_factor, mode="bilinear", align_corners=True))
        fm_concat = torch.cat(fm_upsampled, 1)

        # output conv layers
        output = fm_concat
        for conv_out in self.convs_out:
            output = conv_out(output)
        return output
