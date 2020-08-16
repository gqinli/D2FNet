import torch.nn as nn #导入包
import torch
import numpy as np
from collections import OrderedDict
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, pretrained=True, pretrained_path=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize weights
        if pretrained == True and pretrained_path != None:
            self.load_pretrained(pretrained_path)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=12, dilation=12)  # fc6 in paper
        self.bn6 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1)  # fc7 in paper
        self.bn7 = nn.BatchNorm2d(1024)
        self.relu7 = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        xconv1 = self.conv1(x)
        xbn1 = self.bn1(xconv1)
        xru1 = self.relu(xbn1)
        xpl1 = self.maxpool(xru1)

        xlayer1 = self.layer1(xpl1)
        xlayer2 = self.layer2(xlayer1)
        xlayer3 = self.layer3(xlayer2)
        xlayer4 = self.layer4(xlayer3)

        conv6 = self.conv6(xlayer4)
        bn6 = self.bn6(conv6)
        relu6 = self.relu6(bn6)
        conv7 = self.conv7(relu6)
        bn7 = self.bn7(conv7)
        relu7 = self.relu7(bn7)
        return xpl1, xlayer1, xlayer2, xlayer3, xlayer4, relu7

    def load_pretrained(self, path):
        print('Initialize ResNet from pretrained model.')
        saved_state_dict = torch.load(path)
        new_params = self.state_dict()
        for param in saved_state_dict.keys():
            if not param.startswith('fc'):
                new_params[param] = saved_state_dict[param]
        self.load_state_dict(new_params)


def ResNetOne(pretrained_path): #定义网络为3个block
    return ResNet(BasicBlock, [2, 2, 2, 2], pretrained_path=pretrained_path) #网络的返回值

if __name__ == '__main__':
       print('Test Net')
       data1 = torch.rand(2, 3, 160, 160)
       input_var1 = torch.autograd.Variable(data1)
       resnet50_path = '../resnext/resnet18.pth'
       model = ResNetOne(resnet50_path)
       output = model(input_var1)
       print(output.size())
