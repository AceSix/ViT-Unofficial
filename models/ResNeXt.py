# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\models\ResNext.py
###   @Author: Ziang Liu
###   @Date: 2020-12-25 15:57:40
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-27 19:22:56
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, D),
            nn.ReLU(),
            nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.GroupNorm(8, D),
            nn.ReLU(),
            nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.net(x)
        residual = self.shortcut(x)
        return F.relu(residual + bottleneck)
    
class ResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 5) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.stage_1 = self.block(self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block(self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block(self.stages[2], self.stages[3], 2)
        self.final_aap = nn.AdaptiveMaxPool2d((1,1))
        self.classifier = nn.Linear(self.stages[3], nlabels)
        init.kaiming_normal(self.classifier.weight)


    def block(self, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        blocks = []
        for bottleneck in range(self.block_depth):
            if bottleneck == 0:
                blocks.append(ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                blocks.append(ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor))
        block = nn.Sequential(*blocks)
        return block

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.final_aap(x)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)
        # return F.softmax(self.classifier(x), dim=-1)

    
class ResNeXt_pretrained(nn.Module):
    def __init__(self, nlabels):
        super(ResNeXt_pretrained, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        self.model.fc = nn.Linear(2048, nlabels, bias=True)

    def forward(self, x):
        y = self.model(x)
        return y
        # return F.softmax(self.classifier(x), dim=-1)