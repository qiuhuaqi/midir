import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import conv_block_1, conv_blocks_2, conv_blocks_3


class BaseNet(nn.Module):
    """Deformable registration network with input from image space
    The Network in paper:
    C. Qin et al., “Joint learning of motion estimation and segmentation for cardiac MR image sequences,” MICCAI 2018
    """
    def __init__(self, n_ch=1):
        super(BaseNet, self).__init__()
        self.conv_blocks1 = conv_blocks_2(n_ch, 64)
        self.conv1 = conv_block_1(128, 64)
        self.conv_blocks2 = conv_blocks_2(64, 128, 2)
        self.conv2 = conv_block_1(256, 64)
        self.conv_blocks3 = conv_blocks_3(128, 256, 2)
        self.conv3 = conv_block_1(512, 64)
        self.conv_blocks4 = conv_blocks_3(256, 512, 2)
        self.conv4 = conv_block_1(1024, 64)
        self.conv_blocks5 = conv_blocks_3(512, 512, 2)
        self.conv5 = conv_block_1(1024, 64)

        self.conv_blocks12 = conv_blocks_2(n_ch, 64)
        self.conv_blocks22 = conv_blocks_2(64, 128, 2)
        self.conv_blocks32 = conv_blocks_3(128, 256, 2)
        self.conv_blocks42 = conv_blocks_3(256, 512, 2)
        self.conv_blocks52 = conv_blocks_3(512, 512, 2)

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
            net['flow']: (tensor, shape NxHxWx2) calculated optical flow
            net['wapred_source']: source images warped towards the target images
        """

        # slice source and target images
        net = {}

        net['conv1'] = self.conv_blocks1(source)
        net['conv2'] = self.conv_blocks2(net['conv1'])
        net['conv3'] = self.conv_blocks3(net['conv2'])
        net['conv4'] = self.conv_blocks4(net['conv3'])
        net['conv5'] = self.conv_blocks5(net['conv4'])

        net['conv1s'] = self.conv_blocks12(target)
        net['conv2s'] = self.conv_blocks22(net['conv1s'])
        net['conv3s'] = self.conv_blocks32(net['conv2s'])
        net['conv4s'] = self.conv_blocks42(net['conv3s'])
        net['conv5s'] = self.conv_blocks52(net['conv4s'])

        net['concat1'] = torch.cat((net['conv1'], net['conv1s']), 1)
        net['concat2'] = torch.cat((net['conv2'], net['conv2s']), 1)
        net['concat3'] = torch.cat((net['conv3'], net['conv3s']), 1)
        net['concat4'] = torch.cat((net['conv4'], net['conv4s']), 1)
        net['concat5'] = torch.cat((net['conv5'], net['conv5s']), 1)

        net['out1'] = self.conv1(net['concat1'])
        net['out2'] = self.conv2(net['concat2'])
        net['out3'] = self.conv3(net['concat3'])
        net['out4'] = self.conv4(net['concat4'])
        net['out5'] = self.conv5(net['concat5'])

        net['out2_up'] = F.interpolate(net['out2'], scale_factor=2, mode='bilinear', align_corners=True)
        net['out3_up'] = F.interpolate(net['out3'], scale_factor=4, mode='bilinear', align_corners=True)
        net['out4_up'] = F.interpolate(net['out4'], scale_factor=8, mode='bilinear', align_corners=True)
        net['out5_up'] = F.interpolate(net['out5'], scale_factor=16, mode='bilinear', align_corners=True)

        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])

        net['op_flow'] = torch.tanh(self.conv8(net['comb_2']))

        return net['op_flow']

