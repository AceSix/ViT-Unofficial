# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\models\ViT.py
###   @Author: Ziang Liu
###   @Date: 2020-12-07 19:25:12
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-25 22:56:13
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
# -*- coding: utf-8 -*-

import torch
from torch import nn

from .Blocks import Transformer, pose_embed

class ViT(nn.Module):
    def __init__(self, dim_hid=256, dim_KQ=128, cls_num=10, patch=2, stride=2, depth=4, dropout=0.1):
        super(ViT, self).__init__()

        self.splitNfc = nn.Conv2d(3, dim_hid, patch, stride, 0)
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(Transformer(dim_hid, dim_hid, dim_KQ, dropout))
        self.transformer = nn.Sequential(*self.transformers)
        
        self.cls = nn.Parameter(torch.randn(1,1,dim_hid))
        self.FC = nn.Sequential(
            nn.Linear(dim_hid, cls_num),
            nn.Sigmoid(),
        )
        self.pose_dim = dim_hid
        
    def forward(self, x):  
        y = self.splitNfc(x)
        y = pose_embed(y, self.pose_dim)

        b, c, h, w = y.shape
        y_col = y.reshape(b, c, h*w).transpose(2,1).contiguous()
        
        cls_token = self.cls.repeat([y_col.shape[0],1,1])
        y_col = torch.cat([cls_token, y_col], dim=1)
        y_col = self.transformer(y_col)

        out = self.FC(y_col[:,0,:])
        return out