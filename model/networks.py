import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import conv_block_1, conv_blocks_2, conv_blocks_3
from model.submodules import spatial_transform


class BaseNet(nn.Module):
    """Deformable registration network with input from image space
    The Network in paper:
    C. Qin et al., “Joint learning of motion estimation and segmentation for cardiac MR image sequences,” MICCAI 2018
    """
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv_blocks1 = conv_blocks_2(1, 64)
        self.conv_blocks2 = conv_blocks_2(64, 128, 2)
        self.conv_blocks3 = conv_blocks_3(128, 256, 2)
        self.conv_blocks4 = conv_blocks_3(256, 512, 2)
        self.conv_blocks5 = conv_blocks_3(512, 512, 2)

        self.conv_blocks12 = conv_blocks_2(1, 64)
        self.conv_blocks22 = conv_blocks_2(64, 128, 2)
        self.conv_blocks32 = conv_blocks_3(128, 256, 2)
        self.conv_blocks42 = conv_blocks_3(256, 512, 2)
        self.conv_blocks52 = conv_blocks_3(512, 512, 2)

        self.conv1 = conv_block_1(128, 64)
        self.conv2 = conv_block_1(256, 64)
        self.conv3 = conv_block_1(512, 64)
        self.conv4 = conv_block_1(1024, 64)
        self.conv5 = conv_block_1(1024, 64)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv_block_1(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

    def forward(self, target, source):
        """
        Forward pass function.

        Args:
            target: target image input to the network, Tensor shape (T-1, 1, H, W)
            source: source image input to the network, Tensor shape (T-1, 1, H, W)

        Returns:
            net['dvf']: (Tensor, shape (N, 2, H, W)) calculated optical flow
        """
        net = {}

        # two branches, weights not shared
        net['conv1s'] = self.conv_blocks1(source)
        net['conv2s'] = self.conv_blocks2(net['conv1s'])
        net['conv3s'] = self.conv_blocks3(net['conv2s'])
        net['conv4s'] = self.conv_blocks4(net['conv3s'])
        net['conv5s'] = self.conv_blocks5(net['conv4s'])

        net['conv1t'] = self.conv_blocks12(target)
        net['conv2t'] = self.conv_blocks22(net['conv1t'])
        net['conv3t'] = self.conv_blocks32(net['conv2t'])
        net['conv4t'] = self.conv_blocks42(net['conv3t'])
        net['conv5t'] = self.conv_blocks52(net['conv4t'])

        # concatenate feature maps of the two branches
        net['concat1'] = torch.cat((net['conv1s'], net['conv1t']), 1)
        net['concat2'] = torch.cat((net['conv2s'], net['conv2t']), 1)
        net['concat3'] = torch.cat((net['conv3s'], net['conv3t']), 1)
        net['concat4'] = torch.cat((net['conv4s'], net['conv4t']), 1)
        net['concat5'] = torch.cat((net['conv5s'], net['conv5t']), 1)

        # convolution on the concatenated feature maps
        net['out1'] = self.conv1(net['concat1'])
        net['out2'] = self.conv2(net['concat2'])
        net['out3'] = self.conv3(net['concat3'])
        net['out4'] = self.conv4(net['concat4'])
        net['out5'] = self.conv5(net['concat5'])

        # upsample all to full resolution and concatenate
        net['out2_up'] = F.interpolate(net['out2'], scale_factor=2, mode='bilinear', align_corners=True)
        net['out3_up'] = F.interpolate(net['out3'], scale_factor=4, mode='bilinear', align_corners=True)
        net['out4_up'] = F.interpolate(net['out4'], scale_factor=8, mode='bilinear', align_corners=True)
        net['out5_up'] = F.interpolate(net['out5'], scale_factor=16, mode='bilinear', align_corners=True)
        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)

        # final convolution and output
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])
        net['out'] = self.conv8(net['comb_2'])

        # warp the source image towards target
        warped_source = spatial_transform(source, net['out'])
        return net['out'], warped_source


class SiameseFCN(nn.Module):
    def __init__(self):
        super(SiameseFCN, self).__init__()

        self.conv_blocks1 = conv_blocks_2(1, 64)
        self.conv_blocks2 = conv_blocks_2(64, 128, 2)
        self.conv_blocks3 = conv_blocks_3(128, 256, 2)
        self.conv_blocks4 = conv_blocks_3(256, 512, 2)
        self.conv_blocks5 = conv_blocks_3(512, 512, 2)

        self.conv1 = conv_block_1(128, 64)
        self.conv2 = conv_block_1(256, 64)
        self.conv3 = conv_block_1(512, 64)
        self.conv4 = conv_block_1(1024, 64)
        self.conv5 = conv_block_1(1024, 64)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv_block_1(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

    def forward(self, target, source):
        """
        Forward pass function.

        Args:
            target: target image input to the network, Tensor shape (T-1, 1, H, W)
            source: source image input to the network, Tensor shape (T-1, 1, H, W)

        Returns:
            net['dvf']: (Tensor, shape (N, 2, H, W)) calculated optical flow
        """
        # two branches, shared weights
        conv1s = self.conv_blocks1(source)
        conv2s = self.conv_blocks2(conv1s)
        conv3s = self.conv_blocks3(conv2s)
        conv4s = self.conv_blocks4(conv3s)
        conv5s = self.conv_blocks5(conv4s)

        conv1t = self.conv_blocks1(target)
        conv2t = self.conv_blocks2(conv1t)
        conv3t = self.conv_blocks3(conv2t)
        conv4t = self.conv_blocks4(conv3t)
        conv5t = self.conv_blocks5(conv4t)

        # concatenate feature maps of the two branches
        concat1 = torch.cat((conv1s, conv1t), 1)
        concat2 = torch.cat((conv2s, conv2t), 1)
        concat3 = torch.cat((conv3s, conv3t), 1)
        concat4 = torch.cat((conv4s, conv4t), 1)
        concat5 = torch.cat((conv5s, conv5t), 1)

        # convolution on the concatenated feature maps
        out1 = self.conv1(concat1)
        out2 = self.conv2(concat2)
        out3 = self.conv3(concat3)
        out4 = self.conv4(concat4)
        out5 = self.conv5(concat5)

        # upsample all to full resolution and concatenate
        out2_up = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)
        out3_up = F.interpolate(out3, scale_factor=4, mode='bilinear', align_corners=True)
        out4_up = F.interpolate(out4, scale_factor=8, mode='bilinear', align_corners=True)
        out5_up = F.interpolate(out5, scale_factor=16, mode='bilinear', align_corners=True)
        concat = torch.cat((out1, out2_up, out3_up, out4_up, out5_up), 1)

        # final convolution and output
        comb1 = self.conv6(concat)
        comb2 = self.conv7(comb1)

        out = self.conv8(comb2)
        return out

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#
#     def forward(self, target, source):
#         # warp the source image towards target
#         warped_source = spatial_transform(source, net['out'])
#         return dvf, warped_source


