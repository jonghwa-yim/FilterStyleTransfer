__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import torch.nn as nn
import torch.nn.functional as F


class TransformerNet(nn.Module):
    def __init__(self, aleatoric=False, **kwargs):
        super(TransformerNet, self).__init__()
        self.aleatoric = aleatoric
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        if aleatoric:
            # Heteroscedastic Uncertainty
            self.deconv1_var = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
            self.in4_var = nn.InstanceNorm2d(64, affine=True)
            self.deconv2_var = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
            self.in5_var = nn.InstanceNorm2d(32, affine=True)
            self.deconv3_var = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X, mcdrop=False):
        var = None
        y = self.relu(self.in1(self.conv1(X)))
        y = F.dropout(y, p=0.1, training=mcdrop)
        y = self.relu(self.in2(self.conv2(y)))
        y = F.dropout(y, p=0.1, training=mcdrop)
        y = self.relu(self.in3(self.conv3(y)))
        y = F.dropout(y, p=0.1, training=mcdrop)
        y = self.res1(y, mcdrop)
        y = self.res2(y, mcdrop)
        y = self.res3(y, mcdrop)
        y = self.res4(y, mcdrop)
        y = self.res5(y, mcdrop)
        if self.aleatoric:
            var = self.relu(self.in4_var(self.deconv1_var(y)))
            var = self.relu(self.in5_var(self.deconv2_var(var)))
            var = self.deconv3_var(var)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        if self.aleatoric:
            return y, var
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """

        :rtype: nn.Module
        """
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        # out = F.dropout(out, p=0.3, training=mcdrop)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        """

        :rtype: nn.Module
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x, mcdrop):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = F.dropout(out, p=0.1, training=mcdrop)
        out = self.in2(self.conv2(out))
        out = F.dropout(out, p=0.1, training=mcdrop)
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        """

        :rtype: nn.Module
        """
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = F.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        # out = F.dropout(out, p=0.3, training=mcdrop)
        return out
