# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\models\Blocks.py
###   @Author: Ziang Liu
###   @Date: 2020-12-25 15:57:40
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-25 16:10:20
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import math
import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, version='resnet50'):
        super(ResNetBlock, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.6.0', version, pretrained=True)
        self.net = nn.Sequential(
            *list(resnet.children())[:8]
        )
    def forward(self, x):
        y = self.net(x)
        return y

class ResNet_predictor(nn.Module):
    def __init__(self, cls_num, version='resnet50'):
        super(ResNet_predictor, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', version, pretrained=True)
        self.out = nn.Sequential(
            nn.Linear(1000, cls_num),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.resnet(x)
        y = self.out(y)
        return y

class Transformer(nn.Module):
    def __init__(self, dim_in=3*24*24, dim_hid=256, dim_KQ=128, dropout=0.1):
        super(Transformer, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_KQ = dim_KQ

        self.FC1 = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            nn.LayerNorm(dim_hid),
            nn.ReLU()
        )

        self.KQ_embed = nn.Linear(dim_hid, dim_KQ*2)

        self.FC2 = nn.Sequential(
            nn.Linear(dim_hid, dim_hid),
            nn.LayerNorm(dim_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hid, dim_hid),
            nn.LayerNorm(dim_hid),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.FC1(x)

        # Transformer
        KQ = self.KQ_embed(y)                                  # b, hw, 2c
        K, Q = KQ[:,:,:self.dim_KQ], KQ[:,:,self.dim_KQ:]      # b, hw, c
        att = torch.matmul(Q, K.transpose(-1,-2))
        att = (att/math.sqrt(self.dim_hid)).softmax(-1)
        att = torch.matmul(att, y)+y                           # b, hw, c
        out = self.FC2(att)+att                                # b, hw, c

        return out

def pose_embed(x, dim):
    h,w = x.shape[-2:]
    a = torch.arange(0.0, h, 1.0).unsqueeze(1).repeat(1,w).to(x.device)
    b = torch.arange(0.0, w, 1.0).unsqueeze(0).repeat(h,1).to(x.device)
    pose = torch.stack([a,b], -1).unsqueeze(0).repeat(x.shape[0],1,1,1)      # b, h, w, 2
    
    inv_x = torch.FloatTensor(1 / (1000 ** (torch.arange(0.0, dim, 2.0) /dim) ) ).to(x.device)
    inv_y = torch.FloatTensor(1 / (1000 ** (torch.arange(1.0, dim+1, 2.0) /dim) ) ).to(x.device)
    
    inv = torch.stack([inv_x, inv_y], 0)                            # 2, c/2
    
    out = torch.matmul(pose, inv)                                   # b, h, w, c/2
    out = torch.stack([torch.sin(out), torch.cos(out)], dim=-1)     # b, h, w, c
    out = out.reshape(*out.shape[:-2], -1)
    out = out.permute(0,3,1,2).contiguous()
    return x+out