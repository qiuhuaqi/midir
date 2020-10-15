import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from model.network.base import conv_Nd, avg_pool
from model.network.base import conv_blocks_2, conv_blocks_3, conv_block_1
from utils.misc import param_dim_setup


class UNet(nn.Module):
    """
    Slight modification of the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 dim=2,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 out_channels=(16, 16),
                 conv_before_out=True
                 ):
        super(UNet, self).__init__()

        self.dim = dim

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    conv_Nd(dim, in_ch, enc_channels[i], stride=stride),
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

        # (optional) conv layers before prediction
        if conv_before_out:
            self.out_layers = nn.ModuleList()
            for i in range(len(out_channels)):
                in_ch = dec_channels[-1] + enc_channels[0] if i == 0 else out_channels[i-1]
                self.out_layers.append(
                    nn.Sequential(
                        conv_Nd(dim, in_ch, out_channels[i]),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )

            # final prediction layer with additional conv layers
            self.out_layers.append(
                conv_Nd(dim, out_channels[-1], dim)
            )

        else:

            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                conv_Nd(dim, dec_channels[-1] + enc_channels[0], dim)
            )

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)

        # further convs and prediction
        y = dec_out
        for out_layer in self.out_layers:
            y = out_layer(y)
        return [y]


class FFDNet(nn.Module):
    """
    Encoder/downsample only network specifically for FFD module.
    Work with both 2D and 3D
    """
    def __init__(self,
                 dim=2,
                 img_size=(192, 192),
                 cpt_spacing=(8, 8),
                 enc_channels=(64, 128, 256),
                 out_channels=(128, 64)
                 ):
        super(FFDNet, self).__init__()

        self.dim = dim

        # parameter dimension check
        self.img_size = param_dim_setup(img_size, dim)
        self.cpt_spacing = param_dim_setup(cpt_spacing, dim)

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i-1]
            out_ch = enc_channels[i]
            self.enc.append(nn.Sequential(conv_Nd(dim, in_ch, out_ch),
                                          nn.LeakyReLU(0.2),
                                          conv_Nd(dim, out_ch, out_ch),
                                          nn.LeakyReLU(0.2),
                                          avg_pool(self.dim)
                                          )
                            )

        # 1x1(x1) conv layers before prediction
        self.out_layers = nn.ModuleList()
        for i in range(len(out_channels)):
            in_ch = enc_channels[-1] if i == 0 else out_channels[i-1]
            out_ch = out_channels[i]
            self.out_layers.append(nn.Sequential(conv_Nd(self.dim, in_ch, out_ch, kernel_size=1, padding=0),
                                                 nn.LeakyReLU(0.2))
                                   )

        # final prediction layer
        self.out_layers.append(
            conv_Nd(self.dim, out_channels[-1], self.dim, kernel_size=1, padding=0)
        )

        # determine output size from image size and control point spacing
        self.output_size = tuple([int(isz // cps) + 2
                                  for isz, cps in zip(self.img_size, self.cpt_spacing)])

    def _resize(self, x):
        inter_mode = "bilinear"
        if self.dim == 3:
            inter_mode = "trilinear"
        return F.interpolate(x, self.output_size, mode=inter_mode, align_corners=False)


    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        for enc in self.enc:
            x = enc(x)

        # resize the feature map to match output size
        y = self._resize(x)

        # 1x1(x1) conv and prediction
        for out_layer in self.out_layers:
            y = out_layer(y)
        return [y]


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
