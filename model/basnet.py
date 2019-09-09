import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

try:
    import sys
    sys.path.append('..')
    from .resnet import resnet34, BasicBlock
except:
    from resnet import resnet34, BasicBlock


class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual


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

class BASNet(nn.Module):
    def __init__(self,n_channels,n_classes,activation = None, dr = 0):
        super(BASNet,self).__init__()

        self.activation = activation
        ## -------------Encoder--------------
        self.resnet = resnet34(pretrained=True)
        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        self.decoder = UnetDecoder(encoder_channels=(512, 256, 128, 64, 64), 
                                dropout = dr)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear', align_corners=True)###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear', align_corners=True)
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear', align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)

        ## -------------Side Output--------------
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(256,1,3,padding=1)
        self.outconv4 = nn.Conv2d(128,1,3,padding=1)
        self.outconv3 = nn.Conv2d(64,1,3,padding=1)
        self.outconv2 = nn.Conv2d(32,1,3,padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)


    def forward(self,x):
        # x N*C*512*512
        hx = x

        ## -------------Encoder-------------

        h4, h3, h2, h1, h0 = self.resnet(x)

        ## -------------Bridge-------------
        hx = h4
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))
        ## -------------Decoder-------------

        d1, d2, d3, d4, d5 = self.decoder([hbg, h3, h2, h1, h0])
        # d1: 512, d2: 256, ..., d5: 32
        ## -------------Side Output-------------
        db = self.outconv6(hbg)
        db = self.upscore6(db) # 16->512

        d5 = self.outconv5(d5)
        d5 = self.upscore5(d5) # 32->512

        d4 = self.outconv4(d4)
        d4 = self.upscore4(d4) # 64->512

        d3 = self.outconv3(d3)
        d3 = self.upscore3(d3) # 128->512

        d2 = self.outconv2(d2)
        d2 = self.upscore2(d2) # 256->512

        # # ## -------------Refine Module-------------
        # dout = self.refunet(d1) # 512

        if self.activation is None:
            return d1, d2, d3, d4, d5, db
        else:
            return torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6), torch.sigmoid(db)

if __name__ == "__main__":
    import numpy as np 
    a = torch.ones(1,3,512,512)
    b = BASNet(3,1)
    # c = b(a)
    print(b)
    # print(c[0].shape, c[1].shape, c[2].shape,c[-1].shape)