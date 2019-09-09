import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU
from ..base.model import Model


class PAM(nn.Module):
    __name__ = 'PAM'
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)

        self.conv4 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels)

        self.act = nn.Softmax(dim=-1)

    def forward(self,inputs):
        
        
        net1 = self.conv1(inputs)
        net2 = self.conv2(inputs)
        net3 = self.conv3(inputs)

        b,c,h,w = net1.size()
        net1 = net1.view(b,-1,c).contiguous()
        net2 = net2.view(b,c,-1).contiguous()

        respones = torch.matmul(net1,net2)
        respones = self.act(respones)

        net3 = net3.view(b,-1,c)

        res = torch.matmul(respones,net3)
        res = res.view(b,c,h,w)

        output = self.conv4(res)
        output = self.bn4(output)

        return output + inputs

class CAM(nn.Module):
    __name__ = 'CAM'
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.Softmax(dim=-1)

    def forward(self,inputs):
        
        b,c,h,w = inputs.size()
        net1 = inputs.view(b,c,-1).contiguous()
        net2 = inputs.view(b,-1,c).contiguous()

        respones = torch.matmul(net1,net2)
        respones = self.act(respones)

        net3 = inputs.view(b,c,-1)

        res = torch.matmul(respones,net3)
        res = res.view(b,c,h,w)
        output = self.conv1(res)
        output = self.bn1(output)

        return output + inputs


class non_local_block(nn.Module):

    __name__ = 'pa_ca_non_local_block'
    def __init__(self,in_channels):
        super().__init__()

        self.pam = PAM(in_channels)
        self.cam = CAM(in_channels)

        # self.act = nn.ReLU(inplace=True)
    
    def forward(self,inputs):

        pam = self.pam(inputs)
        cam = self.cam(inputs)

        res = pam + cam

        return res


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            dropout = 0.2
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.drop = nn.Dropout2d(dropout)
        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])

        x = self.drop(x)

        x = self.final_conv(x)

        return x

class Non_local_UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            dropout = 0.2
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        self.att1 = non_local_block(encoder_channels[1])
        self.att2 = non_local_block(encoder_channels[2])
        self.att3 = non_local_block(encoder_channels[3])
        self.att4 = non_local_block(encoder_channels[4])

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.drop = nn.Dropout2d(dropout)
        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        skips[0] = self.att1(skips[0])
        skips[1] = self.att2(skips[1])
        skips[2] = self.att3(skips[2])
        skips[3] = self.att4(skips[3])

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])

        x = self.drop(x)

        x = self.final_conv(x)

        return x

