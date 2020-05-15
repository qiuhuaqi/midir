import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.base import conv_block_1, conv_blocks_2, conv_blocks_3

"""
Networks for DVF transformation model
"""

class SiameseNet(nn.Module):
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
            up_factor = 2 ** (l+1)
            fm_upsampled.append(
                F.interpolate(fm_red, scale_factor=up_factor, mode="bilinear", align_corners=True))
        fm_concat = torch.cat(fm_upsampled, 1)

        # output conv layers
        output = fm_concat
        for conv_out in self.convs_out:
            output = conv_out(output)
        return output


""""""

