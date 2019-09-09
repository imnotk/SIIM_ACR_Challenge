import torch
import torch.nn as nn
import torch.nn.functional as F



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


class non_local(nn.Module):

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
        
class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)