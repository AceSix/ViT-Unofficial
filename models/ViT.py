# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \undefinede:\AI_Lab\ViT-Unofficial\models\ViT.py
###   @Author: Ziang Liu
###   @Date: 2020-12-07 19:25:12
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-29 10:49:08
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
# -*- coding: utf-8 -*-

import torch
from torch import nn

from .Blocks import Transformer, pose_embed

class ViT(nn.Module):
    def __init__(self, dim_in=3, dim_hid=256, cls_num=10, patch=2, stride=2, depth=4, dropout=0.1):
        super(ViT, self).__init__()

        self.splitNfc = nn.Conv2d(dim_in, dim_hid, patch, stride, 0)
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(Transformer(dim_hid, dim_hid, dim_hid, dropout))
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

from .Blocks import Transformer, pose_embed

class ViT_block(nn.Module):
    def __init__(self, patch_size=24, dim_hid=256, dim_KQ=128, cls_num=10, depth=1, dropout=0.1):
        super(ViT_block, self).__init__()

        self.ps = patch_size
        transformers = [Transformer(dim_in=2048*patch_size**2, dim_hid=dim_hid, dim_KQ=dim_KQ, dropout=dropout)]
        for _ in range(depth-1):
            transformers.append(Transformer(dim_in=dim_hid, dim_hid=dim_hid, dim_KQ=dim_KQ, dropout=dropout))
        
        self.transformer = nn.Sequential(*transformers)
        self.cls = nn.Parameter(torch.randn(1,1,2048*self.ps**2))
        self.FC = nn.Sequential(
            nn.Linear(dim_hid, cls_num),
            nn.Sigmoid(),
        )
        
    def forward(self, x):  
        unfold = nn.Unfold(kernel_size=self.ps, stride=self.ps)
        # b, c, h, w
        y = pose_embed(x, 2048)
        y_col = unfold(y).permute(0,2,1)
        cls_token = self.cls.repeat([y_col.shape[0],1,1])
        y_col = torch.cat([cls_token, y_col], dim=1)
        y_col = self.transformer(y_col)
        out = self.FC(y_col[:,0,:])
        return out
        
class ViT_ResNeXt(nn.Module):
    def __init__(self, dim=512, cls_num=10, patch=2, depth=1):
        super(ViT_ResNeXt, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:8]
        )
        self.classifier = ViT_block(patch, dim, dim, cls_num, depth)

    def forward(self, x):
        y = self.backbone(x)
        y = self.classifier(y)
        return y
