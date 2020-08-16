from resnext import resnetBase_182
import torch.nn.functional as F
from resnext.config import resenet18_path
import torch
from torch import nn


class weight_learn(nn.Module):
    def __init__(self, inchan, middle, outchan):
        super(weight_learn, self).__init__()

        self.fc1 = nn.Linear(inchan, middle)
        self.fc2 = nn.Linear(middle, outchan)

    def forward(self, d):
        depth_global = F.avg_pool2d(d, kernel_size=d.shape[-2:])
        depth_global = depth_global.view(depth_global.size(0), -1)
        mid = self.fc1(depth_global)
        weight = self.fc2(mid)
        return weight


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

        return f43210fuse, sideout5, sideout4, sideout3, sideout2


class R3Net(nn.Module):
    def __init__(self, num_class=1):
        super(R3Net, self).__init__()
        self.net1 = resnetBase_182.ResNetOne(resenet18_path)
        self.net2 = resnetBase_182.ResNetOne(resenet18_path)

        # reduce channel for RGB
        self.f7 = self._make_fea(1024, 256, ks=1, pad=0)
        self.f5 = self._make_fea(512, 256, ks=1, pad=0)
        self.f4 = self._make_fea(256, 128, ks=1, pad=0)

        # reduce channel for depth
        self.d7 = self._make_fea(1024, 256, ks=1, pad=0)
        self.d5 = self._make_fea(512, 256, ks=1, pad=0)
        self.pre45 = self._make_fea(256 + 256, 128, ks=1, pad=0)

        # learn weight from depth
        self.weight5 = weight_learn(1024, 128, 1)
        self.weight4 = weight_learn(512, 128, 1)
        self.weight3 = weight_learn(256, 128, 1)
        self.weight2 = weight_learn(128, 128, 1)
        self.weight1 = weight_learn(64, 64, 1)

        # RGB feature Fusion
        self.R5 = self._make_fea(256 * 2, 256, ks=1, pad=0)
        self.R4 = self._make_fea(256 * 2, 128, ks=1, pad=0)
        self.R3 = self._make_fea(256 * 2 + 128, 128, ks=1, pad=0)
        self.R2 = self._make_fea(256 * 2 + 128 + 128, 128, ks=1, pad=0)
        self.R1 = self._make_fea(256 * 2 + 128 + 128 + 64, 128, ks=1, pad=0)

        # FPN fusion
        self.decoder = decoderFPN()

        # prediction
        self.pred = nn.Conv2d(128, num_class, kernel_size=1, padding=0, stride=1, bias=False)
        self.preG = nn.Conv2d(128, num_class, kernel_size=1, padding=0, stride=1, bias=False)

    def _make_fea(self, in_features, out_features, ks=1, pad=0):
        conv = nn.Conv2d(in_features, out_features, kernel_size=ks, padding=pad, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    # def upBlock(self, x, up_scale):
    #     resUp = []
    #     w_ori, h_ori = x.shape[2:]
    #     for q in range(len(up_scale)):
    #         w = w_ori * up_scale[q]
    #         h = h_ori * up_scale[q]
    #         x_up = F.interpolate(x, (int(h), int(w)), mode='bilinear', align_corners=False)
    #         resUp.append(x_up)
    #     return resUp

    def upBlock(self, x, up_scale):
        resUp = []
        for q in range(len(up_scale)):
            size = up_scale[q].size()[2:]
            x_up = F.interpolate(x, size, mode='bilinear', align_corners=False)
            resUp.append(x_up)
        return resUp

    def forward(self, x, depth):
        (conv1, conv2, conv3, conv4, conv5, conv7) = self.net1(x)
        (conv1_depth, conv2_depth, conv3_depth, conv4_depth, conv5_depth, conv7_depth) = self.net2(
            depth)  ## depth final layer

        # reduce channel for RGB
        f7 = self.f7(conv7)
        f5 = self.f5(conv5)
        f4 = self.f4(conv4)
        f3 = conv3
        f2 = conv2

        # prediction salmap from depth
        d5 = self.d5(conv5_depth)
        d7 = self.d7(conv7_depth)
        d77 = F.upsample(d7, size=d5.size()[2:], mode='bilinear', align_corners=False)
        d577 = torch.cat((d5, d77), dim=1)
        pre45 = self.pre45(d577)

        # learn weights from depth
        w5 = self.weight5(conv7_depth)
        w4 = self.weight4(conv5_depth)
        w3 = self.weight3(conv4_depth)
        w2 = self.weight2(conv3_depth)
        w1 = self.weight1(conv2_depth)

        ## upsample each level RGB
        # up5 = self.upBlock(f5, [2, 4, 8])
        # up4 = self.upBlock(f4, [2, 4])
        # up3 = self.upBlock(f3, [2])
        up7 = self.upBlock(f7, [f5, f4, f3, f2])
        up5 = self.upBlock(f5, [f4, f3, f2])
        up4 = self.upBlock(f4, [f3, f2])
        up3 = self.upBlock(f3, [f2])

        ## fuse later layer
        # 1/32, w4
        w5 = w5.unsqueeze(1).unsqueeze(1)
        R5 = torch.cat((f7 * w5, d7 * w5), dim=1)
        R5 = self.R5(R5)  # 256, 1/32

        # 1/16
        w4 = w4.unsqueeze(1).unsqueeze(1)
        R4 = torch.cat((up7[0] * w5, f5 * w4), dim=1)  # 256, 1/16
        R4 = self.R4(R4)

        # 1/8
        w3 = w3.unsqueeze(1).unsqueeze(1)
        R3 = torch.cat((up7[1] * w5, up5[0] * w4, f4 * w3), dim=1)
        R3 = self.R3(R3)  # 256, 1/8

        # 1/4
        w2 = w2.unsqueeze(1).unsqueeze(1)
        R2 = torch.cat((up7[2] * w5, up5[1] * w4, up4[0] * w3, f3 * w2), dim=1)
        R2 = self.R2(R2)  # 128, 1/4

        # 1/4
        w1 = w1.unsqueeze(1).unsqueeze(1)
        R1 = torch.cat((up7[3] * w5, up5[2] * w4, up4[1] * w3, up3[0] * w2, f2 * w1), dim=1)
        R1 = self.R1(R1)  # 128, 1/4

        # FPN fusion
        decoder, sideout5, sideout4, sideout3, sideout2 = self.decoder(R5, R4, R3, R2, R1, x)

        # prediction
        depthMap = self.pred(pre45)
        salMap = self.preG(decoder)

        depthMap = F.upsample(depthMap, size=x.size()[2:], mode='bilinear', align_corners=False)
        salMap = F.upsample(salMap, size=x.size()[2:], mode='bilinear', align_corners=False)

        if self.training:
            return depthMap, salMap, sideout5, sideout4, sideout3, sideout2
        return F.sigmoid(
            salMap)  # ,F.sigmoid(conv1[:,0,:,:]),F.sigmoid(conv2[:,0,:,:]),F.sigmoid(conv3[:,0,:,:]),F.sigmoid(conv4[:,0,:,:]),F.sigmoid(conv5[:,0,:,:])


if __name__ == '__main__':
    print('Test Net')
    data1 = torch.rand(2, 3, 160, 160)
    input_var1 = torch.autograd.Variable(data1)
    model = R3Net()
    output = model(input_var1)
    print(output.size())

