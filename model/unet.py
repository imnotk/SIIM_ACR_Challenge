import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet34
import torch.nn.functional as F

try:
    import sys
    sys.path.append('..')
    from .resnet import resnet34, BasicBlock
    from .block import SCse, non_local
except:
    from resnet import resnet34, BasicBlock
    from block import SCse, non_local



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UnetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            dropout = 0.2
    ):
        super().__init__()


        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.Sc1 = SCse(out_channels[0])
        self.Sc2 = SCse(out_channels[1])
        self.Sc3 = SCse(out_channels[2])
        self.Sc4 = SCse(out_channels[3])
        self.Sc5 = SCse(out_channels[4])

        self.drop = nn.Dropout2d(dropout)

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

        x = self.layer1([encoder_head, skips[0]])
        x0 = x
        x = self.layer2([x, skips[1]])
        x1 = x
        x = self.layer3([x, skips[2]])
        x2 = x
        x = self.layer4([x, skips[3]])
        x3 = x
        x = self.layer5([x, None])

        x = self.drop(x)

        x = self.final_conv(x)

        return x, x3, x2, x1, x0

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,activation = None, weight=None, dr = 0):
        super(UNet,self).__init__()

        self.activation = activation
        ## -------------Encoder--------------
        self.resnet = resnet34(pretrained=True, weight=weight)
        ## -------------Decoder--------------
        self.decoder = UnetDecoder(encoder_channels=(512, 256, 128, 64, 64), 
                                dropout = dr)
        # SCse block
        self.sc4 = SCse(dim=512)
        self.sc3 = SCse(dim=256)
        self.sc2 = SCse(dim=128)
        self.sc1 = SCse(dim=64)


    def forward(self,x):
        # x N*C*512*512
        ## -------------Encoder-------------

        h4, h3, h2, h1, h0 = self.resnet(x)

        h4 = self.sc4(h4)
        h3 = self.sc3(h3)
        h2 = self.sc2(h2)
        h1 = self.sc1(h1)

        ## -------------Decoder-------------
        d1, d2, d3, d4, d5 = self.decoder([h4, h3, h2, h1, h0])
        # d1: 512, d2: 256, ..., d5: 32
    


        if self.activation is None:
            return d1
        else:
            return torch.sigmoid(d1)

if __name__ == "__main__":
    a = UNet(3,1,weight="/4T/Public/zhujian/siim_acr/pretrained/ckpt/resnet34/0_size512.pth")
    d = torch.ones(3,3,768,768)
    c = a(d)
    print(c[0].shape,c[1].shape)