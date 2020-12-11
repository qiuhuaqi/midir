import math
import torch
from torch import nn as nn

from core_modules.network.base import conv_Nd, interpolate_
from utils.misc import param_ndim_setup


class UNet(nn.Module):
    """
    Adpated from the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 out_channels=(16, 16),
                 conv_before_out=True
                 ):
        super(UNet, self).__init__()

        self.ndim = ndim

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    conv_Nd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i-1] + enc_channels[-i-1]
            self.dec.append(
                nn.Sequential(
                    conv_Nd(ndim, in_ch, dec_channels[i], a=0.2),
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
                        conv_Nd(ndim, in_ch, out_channels[i], a=0.2),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )

            # final prediction layer with additional conv layers
            self.out_layers.append(
                conv_Nd(ndim, out_channels[-1], ndim)
            )

        else:

            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                conv_Nd(ndim, dec_channels[-1] + enc_channels[0], ndim)
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


class MultiResUNet(UNet):
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 ml_lvls=3
                 ):
        super(MultiResUNet, self).__init__(ndim=ndim,
                                           enc_channels=enc_channels,
                                           dec_channels=dec_channels,
                                           out_channels=None,
                                           conv_before_out=False)
        # number of multi-resolution levels
        self.ml_lvls = ml_lvls

        # conv layers for multi-resolution predictions (low res to high res)
        delattr(self, 'out_layers')  # remove original single-resolution output layers
        self.ml_out_layers = nn.ModuleList()
        for l in range(self.ml_lvls):
            out_layer_l = conv_Nd(ndim, enc_channels[self.ml_lvls - l - 1] + dec_channels[l - self.ml_lvls], ndim)
            self.ml_out_layers.append(out_layer_l)

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        concat_out_ls = []
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)
            concat_out_ls.append(dec_out)

        # multi-resolution dvf outputs
        ml_y = []
        for l, out_layer_l in enumerate(self.ml_out_layers):
            out_l = out_layer_l(concat_out_ls[l-self.ml_lvls])
            if l == 0:
                # direct prediction for the coarsest level
                y_l = out_l
            else:
                # predict residual and add to the upsampled coarser level prediction
                y_l = out_l + 2.0 * interpolate_(ml_y[-1], scale_factor=2)
            ml_y.append(y_l)

        return ml_y


class CubicBSplineNet(UNet):
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 resize_channels=(32, 32),
                 cps=(5, 5, 5),
                 img_size=(176, 192, 176)
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(CubicBSplineNet, self).__init__(ndim=ndim,
                                              enc_channels=enc_channels,
                                              conv_before_out=False)

        # determine and set output control point sizes from image size and control point spacing
        img_size = param_ndim_setup(img_size, ndim)
        cps = param_ndim_setup(cps, ndim)
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
                                  for imsz, c in zip(img_size, cps)])

        # Network:
        # encoder: same u-net encoder
        # decoder: number of decoder layers / times of upsampling by 2 is decided by cps
        num_dec_layers = 4 - int(math.ceil(math.log2(min(cps))))
        self.dec = self.dec[:num_dec_layers]

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                if num_dec_layers > 0:
                    in_ch = dec_channels[num_dec_layers-1] + enc_channels[-num_dec_layers]
                else:
                    in_ch = enc_channels[-1]
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(conv_Nd(ndim, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # final prediction layer
        delattr(self, 'out_layers')  # remove u-net output layers
        self.out_layer = conv_Nd(ndim, resize_channels[-1], ndim)

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)
        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections
        if len(self.dec) > 0:
            dec_out = fm_enc[-1]
            for i, dec in enumerate(self.dec):
                dec_out = dec(dec_out)
                dec_out = self.upsample(dec_out)
                dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)
        else:
            dec_out = fm_enc

        # resize output of encoder-decoder
        x = interpolate_(dec_out, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        y = self.out_layer(x)
        return [y]


# class MultiResBSplineNet(UNet):
#     def __init__(self,
#                  ndim,
#                  ml_lvls=3,
#                  enc_channels=(16, 32, 32, 32, 32),
#                  dec_channels=(32, 32),
#                  resize_channels=(32, 32),
#                  cps=(4, 4, 4),
#                  order=3,
#                  img_size=(176, 192, 176)
#                  ):
#         super(MultiResBSplineNet, self).__init__(ndim=ndim,
#                                                  enc_channels=enc_channels,
#                                                  conv_before_out=False)
#         self.ml_lvls = ml_lvls
#         img_size = param_ndim_setup(img_size, ndim)
#         cps = param_ndim_setup(cps, ndim)
#
#         # calculate the control point parameter output size for the coarsest levels
#         cps_lvl1 = [c * (2 ** self.ml_lvls) for c in cps]
#         img_size_lvl1 = [imsz // 2 for imsz in img_size]
#         self.output_size_lvl1 = tuple([int(imsz // c) + 2 for imsz, c in zip(img_size_lvl1, cps_lvl1)])
#
#         self.output_size_lvl1 = [int(math.ceil(imsz / c)) + order - 1
#                                  for imsz, c in zip(img_size_lvl1, cps_lvl1)]
#         self.output_size_lvl1 = tuple(self.output_size_lvl1)
#
#         # TODO: set the output size for each level (after subdivision) if known,
#         #  otherwise leave it in forward()
#
#
#         # TODO: b-spline subdivision module
#         self.bspline_subdiv = None
#
#
#         # Network:
#         # same encoder
#         # resize layers
#         self.resize_conv = nn.ModuleList()
#         for i in range(resize_channels):
#             in_ch = enc_channels[-1] if i == 0 else resize_channels[i-1]
#             out_ch = resize_channels[i]
#             self.resize_conv.append(nn.Sequential(conv_Nd(ndim, in_ch, out_ch),
#                                                   nn.LeakyReLU(0.2)))
#
#         # decoder (upsample+conv) layers
#         assert len(dec_channels) == self.ml_lvls - 1, \
#             f"Too many decoder layers ({len(dec_channels)}) for the number of resolution ({self.ml_lvls})"
#
#         # TODO: this should be before resizing?
#         self.dec = nn.ModuleList()
#         for i in range(len(dec_channels)):
#             in_ch = resize_channels[-1] if i == 0 else dec_channels[i-1]
#             out_ch = dec_channels[i]
#             self.dec.append(
#                 nn.Sequential(
#                     conv_Nd(ndim, in_ch, out_ch),
#                     nn.LeakyReLU(0.2)
#                 )
#             )
#
#         # multi-resolution prediction layers
#         delattr(self, 'out_layers')  # remove the U-net output layers
#         self.ml_pred_convs = nn.ModuleList()
#         for l in range(self.ml_lvls):
#             in_ch = resize_channels[-1] if l == 0 else dec_channels[l-1]
#             self.ml_pred_convs.append(conv_Nd(ndim, in_ch, ndim))
#
#     def forward(self, tar, src):
#         x = torch.cat((tar, src), dim=1)
#
#         # encode
#         for enc in self.enc:
#             x = enc(x)
#
#         # resize
#         x = self.resize_fn(x, self.output_size_lvl1)
#
#         # decode
#         dec_outs = [x]
#         for dec in self.dec:
#             dec_out = dec(dec_outs[-1])
#             dec_out = self.upsample(dec_out)
#             # TODO: could switch to resize
#             dec_outs.append(dec_out)
#
#         # multi-level prediction layers
#         ml_y = []
#         for l, pred_conv in enumerate(self.ml_pred_convs):
#             pred_l = pred_conv(dec_outs[l])
#             if l == 0:
#                 # direct prediction for the coarsest level
#                 y_l = pred_l
#             else:
#                 # predict residual and add to the upsampled coarser level prediction
#                 # TODO: subdivision instead of interpolation
#                 # TODO: check if scaling is needed (could be done in subdivison already)
#                 y_l =  pred_l + 2.0 * self.bspline_subdiv(ml_y[-1])
#             ml_y.append(y_l)
#
#         return ml_y


