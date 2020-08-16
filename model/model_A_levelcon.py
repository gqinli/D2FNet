from resnext import resnetBase_181
import torch.nn.functional as F
from resnext.config import resenet18_path
import torch
from torch import nn

class decoderFPN(nn.Module):
    def __init__(self):
        super(decoderFPN, self).__init__()
        # decoder: bottom branch for the layer features using FPN
        self.f43 = self.make_fea(256 + 256, 256, ks=3, pad=1)
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
        catf43 = torch.cat((x4, x3), 1)
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
        self.net1 = resnetBase_181.ResNetOne(resenet18_path)
        self.net2 = resnetBase_181.ResNetOne(resenet18_path)

        # RGB feature Fusion
        self.concat7 = self._make_fea(1024 * 2, 256, ks=1, pad=0)
        self.concat5 = self._make_fea(512*2, 256, ks=1, pad=0)
        self.concat4 = self._make_fea(256*2, 128, ks=1, pad=0)
        self.concat3 = self._make_fea(128*2, 128, ks=1, pad=0)
        self.concat2 = self._make_fea(64*2, 128, ks=1, pad=0)
 
        # FPN fusion
        self.decoder = decoderFPN()

        # prediction
        self.preG = nn.Conv2d(128, num_class, kernel_size=1, padding=0, stride=1, bias=False)

    def _make_fea(self, in_features, out_features, ks=1, pad=0):
        conv = nn.Conv2d(in_features, out_features, kernel_size=ks, padding=pad, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, x, depth):
        (conv1, conv2, conv3, conv4, conv5, conv7) = self.net1(x)
        (conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth, conv7_depth) = self.net2(depth)  ## depth final layer

        # rgb and depth each level cat
        R2 = torch.cat((conv2, conv2_depth), 1)
        R2 = self.concat2(R2)
        R3 = torch.cat((conv3, conv3_depth), 1)
        R3 = self.concat3(R3)
        R4 = torch.cat((conv4, conv4_depth), 1)
        R4 = self.concat4(R4)
        R5 = torch.cat((conv5, conv5_depth), 1)
        R5 = self.concat5(R5)
        R7 = torch.cat((conv7, conv7_depth), 1)
        R7 = self.concat7(R7)

        # FPN fusion
        decoder,sideout5,sideout4,sideout3,sideout2 = self.decoder(R7, R5, R4, R3, R2, x)

        # prediction
        salMap = self.preG(decoder)
        salMap = F.upsample(salMap, size=x.size()[2:], mode='bilinear', align_corners=False)

        if self.training:
            return salMap,sideout5,sideout4,sideout3,sideout2
        return F.sigmoid(salMap)


if __name__ == '__main__':
       print('Test Net')
       data1 = torch.rand(2, 3, 160, 160)
       input_var1 = torch.autograd.Variable(data1)
       model = R3Net()
       output = model(input_var1)
       print(output.size())

