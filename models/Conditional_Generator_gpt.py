# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GPT-ST\components\Conditional_Generator_gpt.py
###   @Author: Chen Xuanhong
###   @Date: 2021-06-29 09:33:17
###   @LastEditors: AceSix
###   @LastEditTime: 2021-06-29 10:47:23
###   @Copyright (C) 2021 SJTU. All rights reserved.
###################################################################

import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from components.ResBlock import ResBlock
from components.DeConv   import DeConv
from components.GPT import GPT_Spatial
from components.Conditional_ResBlock import Conditional_ResBlock

class Generator(nn.Module):
    def __init__(
                self, chn=32,
                k_size=3,
                res_num = 5,
                class_num = 3
                ):
        super().__init__()
        padding_size = int((k_size -1)/2)
        self.resblock_list = []
        self.n_class    = class_num
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size=k_size, stride=1, padding=1, bias= False),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size=k_size, stride=2, padding=1,bias =False), # 
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            nn.LeakyReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels = chn*2, out_channels = chn * 4, kernel_size=k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn*4  , out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.LeakyReLU()
        )

        fea_size, res_dim = 512//16, chn * 8
        self.conditional_GPT = GPT_Spatial(fea_size, res_dim, res_num, class_num)

        self.decoder1 = nn.Sequential(
            DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.LeakyReLU(),
            DeConv(in_channels = chn * 8, out_channels = chn *4, kernel_size=k_size),
            nn.InstanceNorm2d(chn *4, affine=True, momentum=0),
            nn.LeakyReLU(),
            DeConv(in_channels = chn * 4, out_channels = chn * 2 , kernel_size=k_size),
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            nn.LeakyReLU()
        )

        self.decoder2 = nn.Sequential(
            DeConv(in_channels = chn *2, out_channels = chn, kernel_size=k_size),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn, out_channels =3, kernel_size=k_size, stride=1, padding=1,bias =True),
            nn.Tanh()
        )

        self.__weights_init__()

    def __weights_init__(self):
        for layer in self.encoder1:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        for layer in self.encoder2:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, input, condition=None, get_feature=False):
        x2 = self.encoder1(input)
        feature = self.encoder2(x2)
        if get_feature:
            return feature
        x5 = self.conditional_GPT(feature, condition)
        out = self.decoder1(x5)
        out = self.decoder2(out)
        return out,feature