"""FlowNet networks"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.correlation_package.correlation import Correlation
# todo: add these back!
from model.submodules import conv, deconv, predict_flow, spatial_transform


# --- FlowNetC --- #
class FlowNetC(nn.Module):
    """Original FlowNetC following NVIDIA's implementation"""

    def __init__(self, batchNorm=True, div_flow=20):
        super(FlowNetC, self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.conv1   = conv(self.batchNorm,   1,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)

        self.corr = Correlation(pad_size=10, kernel_size=1, max_displacement=10, stride1=1, stride2=1, corr_multiply=1)
        self.conv_redir  = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)
        self.deconv1 = deconv(194, 32)
        self.deconv0 = deconv(34, 16)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(34)
        self.predict_flow0 = predict_flow(18)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        # initialise parameters in the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x1, x2):

        # DOWN
        # top input stream
        out_conv1a = self.conv1(x1)  # 64, 1/2
        out_conv2a = self.conv2(out_conv1a)  # 128, 1/4
        out_conv3a = self.conv3(out_conv2a)  # 256, 1/8

        # bottom input stream
        out_conv1b = self.conv1(x2)  # 64, 1/2
        out_conv2b = self.conv2(out_conv1b)  # 128, 1/4
        out_conv3b = self.conv3(out_conv2b)  # 256, 1/8

        # correlation layer
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)  # 441, 1/8

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)  # 32, 1/8
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)  # 473, 1/8

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)  # 256, 1/8
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))  # 512, 1/16
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # 512, 1/32

        # final flow prediction
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # 1024, 1/64
        flow6 = torch.tanh(self.predict_flow6(out_conv6))  # 2, 1/64

        # UP from 1/64
        # transposed convolution (deconv) at predict flow at every upsample
        # 1/32
        flow6_up = self.upsampled_flow6_to_5(flow6)  # 2, 1/32
        out_deconv5 = self.deconv5(out_conv6)  # 512, 1/32
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)   # 512 + 2 + 512 = 1026, 1/32
        flow5 = torch.tanh(self.predict_flow5(concat5))  # 2, 1/32

        # 1/16
        flow5_up = self.upsampled_flow5_to_4(flow5)  # 2, 1/16
        out_deconv4 = self.deconv4(concat5)  # 256, 1/16
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # 256 + 2 + 512 = 770, 1/16
        flow4 = torch.tanh(self.predict_flow4(concat4))  # 2, 1/16

        # 1/8
        flow4_up = self.upsampled_flow4_to_3(flow4)  # 2, 1/8
        out_deconv3 = self.deconv3(concat4)  # 128, 1/8
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)  # 128 + 2 + 256 = 386, 1/8
        flow3 = torch.tanh(self.predict_flow3(concat3))  # 2, 1/8

        # 1/4
        flow3_up = self.upsampled_flow3_to_2(flow3)  # 2, 1/4
        out_deconv2 = self.deconv2(concat3)  # 64, 1/4
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)  # 128 + 64 + 2 = 194, 1/4
        flow2 = torch.tanh(self.predict_flow2(concat2))  # 2, 1/4

        # 1/2
        # not taking anymore downsample stream information from this point
        flow2_up = self.upsampled_flow2_to_1(flow2)  # 2, 1/2
        out_deconv1 = self.deconv1(concat2)  # 32, 1/2
        concat1 = torch.cat((out_deconv1, flow2_up), 1)  # 32 + 2 = 34, 1/2
        flow1 = torch.tanh(self.predict_flow1(concat1))  # 2, 1/2

        # 1/1: full resolution
        flow1_up = self.upsampled_flow1_to_0(flow1)  # 2, 1/1
        out_deconv0 = self.deconv0(concat1)  # 16, 1/1
        concat0 = torch.cat((out_deconv0, flow1_up), 1)  # 16 + 2 = 18, 1/1
        flow0 = torch.tanh(self.predict_flow0(concat0))  # 2, 1/1

        # collect them'all
        flows = [flow0, flow1, flow2, flow3, flow4, flow5, flow6]

        if self.training:
            return flows
        else:
            return flow0



class FlowNetCHighRes(nn.Module):
    """High resolution version of FlowNetC. Maximum downsample 1/16"""
    def __init__(self, batchNorm=True, div_flow=20):
        super(FlowNetCHighRes, self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.conv1   = conv(self.batchNorm,   1,   64, kernel_size=3, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=3, stride=2)

        self.corr = Correlation(pad_size=10, kernel_size=1, max_displacement=10, stride1=1, stride2=1, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv_redir  = conv(self.batchNorm, 128,   32, kernel_size=1, stride=1)

        self.conv2_1 = conv(self.batchNorm, 32+441, 128)

        self.conv3   = conv(self.batchNorm, 128,  256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv3_2 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv4_2 = conv(self.batchNorm, 512,  512)


        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256+256+2, 128)
        self.deconv1 = deconv(128+128+2, 64)
        self.deconv0 = deconv(64+64+2, 32)


        self.predict_flow4 = predict_flow(512)
        self.predict_flow3 = predict_flow(256+256+2)
        self.predict_flow2 = predict_flow(128+128+2)
        self.predict_flow1 = predict_flow(64+64+2)
        self.predict_flow0 = predict_flow(32+2)


        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        # initialise parameters in the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x1, x2):

        # DOWN
        # top input stream
        out_conv1a = self.conv1(x1)  # 64, 1/2
        out_conv2a = self.conv2(out_conv1a)  # 128, 1/4

        # bottom input stream
        out_conv1b = self.conv1(x2)  # 64, 1/2
        out_conv2b = self.conv2(out_conv1b)  # 128, 1/4

        # correlation layer
        out_corr = self.corr(out_conv2a, out_conv2b)
        out_corr = self.corr_activation(out_corr)  # 441, 1/4

        # Redirect top input stream and concatenate with corr op
        out_conv_redir = self.conv_redir(out_conv2a)  # 32, 1/4
        in_conv2_1 = torch.cat((out_conv_redir, out_corr), 1)  # 473, 1/4

        out_conv2_1 = self.conv2_1(in_conv2_1)  # 128, 1/4
        out_conv3 = self.conv3_2(self.conv3_1(self.conv3(out_conv2_1)))  # 256, 1/8
        out_conv4 = self.conv4_2(self.conv4_1(self.conv4(out_conv3)))  # 512, 1/16

        flow4 = torch.tanh(self.predict_flow4(out_conv4))  # 2, 1/16

        # UP from 1/16
        # transposed convolution (deconv) at predict flow at every upsample
        # 1/8
        flow4_up = self.upsampled_flow4_to_3(flow4)  # 2, 1/8
        out_deconv3 = self.deconv3(out_conv4)  # 256, 1/8
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)   # 256 + 256 + 2, 1/8
        flow3 = torch.tanh(self.predict_flow3(concat3))  # 2, 1/8

        # 1/4
        flow3_up = self.upsampled_flow3_to_2(flow3)  # 2, 1/4
        out_deconv2 = self.deconv2(concat3)  # 128, 1/4
        concat2 = torch.cat((out_conv2_1, out_deconv2, flow3_up), 1)  # 128 + 128 + 2, 1/4
        flow2 = torch.tanh(self.predict_flow2(concat2))  # 2, 1/4

        # 1/2
        flow2_up = self.upsampled_flow2_to_1(flow2)  # 2, 1/2
        out_deconv1 = self.deconv1(concat2)  # 64, 1/2
        concat1 = torch.cat((out_conv1a, out_deconv1, flow2_up), 1)  # 64 + 64 + 2, 1/2
        flow1 = torch.tanh(self.predict_flow1(concat1))  # 2, 1/2

        # 1/1
        flow1_up = self.upsampled_flow1_to_0(flow1)  # 2, 1/1
        out_deconv0 = self.deconv0(concat1)  # 32, 1/1
        concat0 = torch.cat((out_deconv0, flow1_up), 1)  # 32 + 2, 1/1
        flow0 = torch.tanh(self.predict_flow0(concat0))

        # collect them'all
        flows = [flow0, flow1, flow2, flow3, flow4]

        if self.training:
            return flows
        else:
            return flow0


class FlowNetCIntern(nn.Module):
    """Modify FlowNetC to get a good model"""

    def __init__(self, pre_corr=False, batchNorm=True, div_flow=20):
        super(FlowNetCIntern, self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.pre_corr = pre_corr

        self.conv1   = conv(self.batchNorm,   1,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)

        self.corr = Correlation(pad_size=10, kernel_size=1, max_displacement=10, stride1=1, stride2=1, corr_multiply=1)
        self.conv_redir  = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)

        self.deconv4 = deconv(512,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)
        self.deconv1 = deconv(194, 32)
        self.deconv0 = deconv(34, 16)

        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(34)
        self.predict_flow0 = predict_flow(18)

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        # initialise parameters in the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x1, x2):

        # DOWN
        # top input stream
        out_conv1a = self.conv1(x1)  # 64, 1/2
        out_conv2a = self.conv2(out_conv1a)  # 128, 1/4
        out_conv3a = self.conv3(out_conv2a)  # 256, 1/8

        # bottom input stream
        out_conv1b = self.conv1(x2)  # 64, 1/2
        out_conv2b = self.conv2(out_conv1b)  # 128, 1/4
        out_conv3b = self.conv3(out_conv2b)  # 256, 1/8

        # correlation layer
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)  # 441, 1/8

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)  # 32, 1/8
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)  # 473, 1/8

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)  # 256, 1/8
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))  # 512, 1/16

        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # 1024, 1/32
        flow5 = torch.tanh(self.predict_flow5(out_conv5))  # 2, 1/32

        # UP from 1/32
        # transposed convolution (deconv) at predict flow at every upsample
        # 1/16
        flow5_up = self.upsampled_flow5_to_4(flow5)  # 2, 1/16
        out_deconv4 = self.deconv4(out_conv5)  # 256, 1/16
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # 512 + 256 + 2 = 770, 1/16
        flow4 = torch.tanh(self.predict_flow4(concat4))  # 2, 1/16

        # 1/8
        flow4_up = self.upsampled_flow4_to_3(flow4)  # 2, 1/8
        out_deconv3 = self.deconv3(concat4)  # 128, 1/8
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)  # 128 + 2 + 256 = 386, 1/8
        flow3 = torch.tanh(self.predict_flow3(concat3))  # 2, 1/8

        # 1/4
        flow3_up = self.upsampled_flow3_to_2(flow3)  # 2, 1/4
        out_deconv2 = self.deconv2(concat3)  # 64, 1/4
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)  # 128 + 64 + 2 = 194, 1/4
        flow2 = torch.tanh(self.predict_flow2(concat2))  # 2, 1/4

        # 1/2
        # not taking anymore downsample stream information from this point
        flow2_up = self.upsampled_flow2_to_1(flow2)  # 2, 1/2
        out_deconv1 = self.deconv1(concat2)  # 32, 1/2
        concat1 = torch.cat((out_deconv1, flow2_up), 1)  # 32 + 2 = 34, 1/2
        flow1 = torch.tanh(self.predict_flow1(concat1))  # 2, 1/2

        # 1/1: full resolution
        flow1_up = self.upsampled_flow1_to_0(flow1)  # 2, 1/1
        out_deconv0 = self.deconv0(concat1)  # 16, 1/1
        concat0 = torch.cat((out_deconv0, flow1_up), 1)  # 16 + 2 = 18, 1/1
        flow0 = torch.tanh(self.predict_flow0(concat0))  # 2, 1/1

        # collect them'all
        flows = [flow0, flow1, flow2, flow3, flow4, flow5]

        if self.training:
            return flows
        else:
            return flow0


# --- FlowNetC --- #
class FlowNetS(nn.Module):
    def __init__(self, input_channels=2, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=3, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.deconv1 = deconv(128+64+2, 32)
        self.deconv0 = deconv(64+32+2, 16)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(64+32+2)
        self.predict_flow0 = predict_flow(16+2)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), 1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = torch.tanh(self.predict_flow6(out_conv6))

        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = torch.tanh(self.predict_flow5(concat5))

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = torch.tanh(self.predict_flow4(concat4))

        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = torch.tanh(self.predict_flow3(concat3))

        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = torch.tanh(self.predict_flow2(concat2))

        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = torch.tanh(self.predict_flow1(concat1))

        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_deconv0, flow1_up), 1)
        flow0 = torch.tanh(self.predict_flow0(concat0))

        flows = [flow0, flow1, flow2, flow3, flow4, flow5, flow6]

        if self.training:
            return flows
        else:
            return flow0


class FlowNetSIntern(nn.Module):
    """FlowNetS model with one less downsample layer,
    max downsample 1/32"""
    def __init__(self, input_channels=2, batchNorm=True):
        super(FlowNetSIntern, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=3, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)

        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.deconv1 = deconv(128+64+2, 32)
        self.deconv0 = deconv(64+32+2, 16)

        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(64+32+2)
        self.predict_flow0 = predict_flow(16+2)

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), 1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        flow5 = self.predict_flow5(out_conv5)
        flow5 = torch.tanh(flow5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4 = torch.tanh(flow4)

        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3 = torch.tanh(flow3)

        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        flow2 = torch.tanh(flow2)

        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)
        flow1 = torch.tanh(flow1)

        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_deconv0, flow1_up), 1)
        flow0 = self.predict_flow0(concat0)
        flow0 = torch.tanh(flow0)

        flows = [flow0, flow1, flow2, flow3, flow4, flow5]

        if self.training:
            return flows
        else:
            return flow0

