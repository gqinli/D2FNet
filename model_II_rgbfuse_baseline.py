from resnext import resnetBase_182
import torch.nn.functional as F
from resnext.config import resenet18_path
import torch
from torch import nn

class decoderFPN(nn.Module):
    def __init__(self):
        super(decoderFPN, self).__init__()
        # decoder: bottom branch for the layer features using FPN
        self.f43 = self.make_fea(256 + 128, 256, ks=3, pad=1)
        self.f432 = self.make_fea(256 + 128, 128, ks=3, pad=1)
        self.f4321 = self.make_fea(128 + 128, 128, ks=3, pad=1)
        self.f43210 = self.make_fea(128 + 128, 128, ks=3, pad=1)
        self.pred4 = nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1, bias=False)
        self.pred43 = nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1, bias=False)
        self.pred432 = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=False)
        self.pred4321 = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=False)

    def make_fea(self, in_features, out_features, ks=3, pad=1):
        conv = nn.Conv2d(in_features, out_features, kernel_size=ks, padding=pad, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU()
        return nn.Sequential(conv, bn, relu)

    def forward(self, x4, x3, x2, x1, x0, x):
        f4_resize = F.upsample(input=x4, size=x3.size()[2:], mode='bilinear', align_corners=False)
        catf43 = torch.cat((f4_resize, x3), 1)
        f43fuse = self.f43(catf43)

        f43_resize = F.upsample(input=f43fuse, size=x2.size()[2:], mode='bilinear', align_corners=False)
        catf432 = torch.cat((f43_resize, x2), 1)
        f432fuse = self.f432(catf432)

        f432_resize = F.upsample(input=f432fuse, size=x1.size()[2:], mode='bilinear', align_corners=False)
        catf4321 = torch.cat((f432_resize, x1), 1)
        f4321fuse = self.f4321(catf4321)

        f4321_resize = F.upsample(input=f4321fuse, size=x0.size()[2:], mode='bilinear', align_corners=False)
        catf43210 = torch.cat((f4321_resize, x0), 1)
        f43210fuse = self.f43210(catf43210)
        
        pref4 = self.pred4(x4)
        sideout5 = F.upsample(pref4, size=x.size()[2:], mode='bilinear', align_corners=False)
        pref43 = self.pred43(f43fuse)
        sideout4 = F.upsample(pref43, size=x.size()[2:], mode='bilinear', align_corners=False)
        pref432 = self.pred432(f432fuse)
        sideout3 = F.upsample(pref432, size=x.size()[2:], mode='bilinear', align_corners=False)
        pref4321 = self.pred4321(f4321fuse)
        sideout2 = F.upsample(pref4321, size=x.size()[2:], mode='bilinear', align_corners=False)

        return f43210fuse,sideout5,sideout4,sideout3,sideout2

class R3Net(nn.Module):
    def __init__(self, num_class=1):
        super(R3Net, self).__init__()
        self.net1 = resnetBase_182.ResNetOne(resenet18_path)

        # reduce channel for RGB
        self.f7 = self._make_fea(1024, 256, ks=1, pad=0)
        self.f5 = self._make_fea(512, 256, ks=1, pad=0)
        self.f4 = self._make_fea(256, 128, ks=1, pad=0)

        # RGB feature Fusion
        self.R5 = self._make_fea(256, 256, ks=1, pad=0)
        self.R4 = self._make_fea(256 * 2, 128, ks=1, pad=0)
        self.R3 = self._make_fea(256 * 2 + 128, 128, ks=1, pad=0)
        self.R2 = self._make_fea(256 * 2 + 128 + 128, 128, ks=1, pad=0)
        self.R1 = self._make_fea(256 * 2 + 128 + 128 + 64, 128, ks=1, pad=0)

        # FPN fusion
        self.decoder = decoderFPN()

        # prediction
        self.preG = nn.Conv2d(128, num_class, kernel_size=1, padding=0, stride=1, bias=False)

    def _make_fea(self, in_features, out_features, ks=1, pad=0):
        conv = nn.Conv2d(in_features, out_features, kernel_size=ks, padding=pad, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def upBlock(self, x, up_scale):
        resUp = []
        for q in range(len(up_scale)):
            size = up_scale[q].size()[2:]
            x_up = F.interpolate(x, size, mode='bilinear', align_corners=False)
            resUp.append(x_up)
        return resUp

    def forward(self, x, depth):
        (conv1, conv2, conv3, conv4, conv5,conv7) = self.net1(x)

        # reduce channel for RGB
        f7 = self.f7(conv7)
        f5 = self.f5(conv5)
        f4 = self.f4(conv4)
        f3 = conv3
        f2 = conv2

        ## upsample each level RGB
        up7 = self.upBlock(f7, [f5, f4, f3, f2])
        up5 = self.upBlock(f5, [f4, f3, f2])
        up4 = self.upBlock(f4, [f3, f2])
        up3 = self.upBlock(f3, [f2])

        ## fuse later layer
        # 1/32, w4
        R5 = self.R5(f7)  # 256, 1/32

        # 1/16
        R4 = torch.cat((up7[0], f5), dim=1)  # 256, 1/16
        R4 = self.R4(R4)

        # 1/8
        R3 = torch.cat((up7[1], up5[0], f4), dim=1)
        R3 = self.R3(R3)  # 256, 1/8

        # 1/4
        R2 = torch.cat((up7[2], up5[1], up4[0], f3), dim=1)
        R2 = self.R2(R2)  # 128, 1/4

        # 1/4
        R1 = torch.cat((up7[3], up5[2], up4[1], up3[0], f2), dim=1)
        R1 = self.R1(R1)  # 128, 1/4

        # FPN fusion
        decoder,sideout5,sideout4,sideout3,sideout2 = self.decoder(R5, R4, R3, R2, R1, x)

        # prediction
        salMap = self.preG(decoder)

        salMap = F.upsample(salMap, size=x.size()[2:], mode='bilinear', align_corners=False)

        if self.training:
            return salMap, sideout5,sideout4,sideout3,sideout2
        return F.sigmoid(salMap)#,F.sigmoid(sideout5),F.sigmoid(sideout4),F.sigmoid(sideout3),F.sigmoid(sideout2)


if __name__ == '__main__':
    print('Test Net')
    data1 = torch.rand(2, 3, 160, 160)
    input_var1 = torch.autograd.Variable(data1)
    model = R3Net()
    output = model(input_var1)
    print(output.size())

