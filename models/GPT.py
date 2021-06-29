# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \ASMegaGAN\components\GPT.py
###   @Author: Ziang Liu
###   @Date: 2021-06-28 23:56:35
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2021-06-29 11:20:34
###   @Copyright (C) 2021 SJTU. All rights reserved.
###################################################################
import torch
from torch import nn
import numpy as np
from components.DeConv import DeConv

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.kqv_embed = nn.Linear(dim, dim*3)
        self.attn = nn.MultiheadAttention(dim, heads)
        self.out_layer = nn.Sequential(
                nn.LayerNorm(dim),
                nn.ReLU()
            )
        self.dim = dim
        
    def forward(self, x, mask):
        
        dim = self.dim
        kqv = self.kqv_embed(x)
        k,q,v = kqv[..., :dim], kqv[..., dim:dim*2], kqv[..., dim*2:]
        out = self.attn(k,q,v, attn_mask=mask)[0]
        out = self.out_layer(out)
        out = out
        return out
    

class GPT_Spatial(nn.Module):
    def __init__(self, feature_size=32, dim=1024, layer_num=4, style_num=9, max_len=16):
        super().__init__()
        assert feature_size%max_len==0, 'feature map cannot be evenly split'
        patch_size = feature_size//max_len
        self.ps = patch_size
        self.max_len = max_len

        self.P_embedding = nn.Conv2d(dim, dim, patch_size, patch_size, 0)
        self.P_deconv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, patch_size, 2)
        )
        self.S_embedding = nn.Embedding(style_num, dim)
        self.X_embedding = nn.Embedding(max_len, dim)
        self.Y_embedding = nn.Embedding(max_len, dim)
        self.XY_merge = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU()
        )
        self.rel_merge = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, 1),
        )
        self.cross_merge = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, 1),
        )
        self.attn_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.attn_layers.append(Attention(dim, 8))
            self.fc_layers.append(nn.Sequential(
                nn.Linear(dim, dim//2),
                nn.Dropout(0.5),
                nn.Linear(dim//2, dim),
                nn.ReLU()
            ))

    def __rel_position(self, pose_embed):
        L, c = pose_embed.shape
        A,B = pose_embed[:,None,:], pose_embed[None,:,:]
        AB = torch.cat([A.repeat(1,L,1), B.repeat(L,1,1)], -1)
        ori2ori = self.rel_merge(AB).squeeze()
        ori2new = self.cross_merge(AB).squeeze()
        new_rel = (-torch.ones(L, 2*L)*1e-9).to(pose_embed.device)
        ori_rel = torch.cat([ori2ori, ori2new], 1)
        rel_pose = torch.cat([ori_rel, new_rel], 0)
        return rel_pose

    def __gen_token(self, style):
        style_emb = self.S_embedding(style)
        style_emb = style_emb[:,None,:]
        style_emb = style_emb.repeat(1,self.max_len*self.max_len, 1)
        return style_emb

    def __gen_pose(self, h, w, device):
        X = torch.arange(0, w)[None, :, None].to(device)
        Y = torch.arange(0, h)[:, None, None].to(device)
        X,Y = X.repeat(h,1,1), Y.repeat(1,w,1)
        X,Y = self.X_embedding(X), self.Y_embedding(Y), 
        XY = torch.cat([X,Y], -1) # (h, w, 2c)
        XY = XY.reshape(h*w, -1)
        XY_emb = self.XY_merge(XY) # (h*w, c)
        return XY_emb

    def forward(self, x, style):
        
        patches = self.P_embedding(x)     # (n, c, h_num, w_num)
        tokens = self.__gen_token(style)     # (n, h_num*w_num, c)
        b, c, h, w = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (n, h_num*w_num, c)
        
        pose_embed = self.__gen_pose(h, w, x.device) # (h_num*w_num, c)
        tokens = tokens + pose_embed[None, :, :].repeat(b, 1, 1)
        patches = patches + pose_embed[None, :, :].repeat(b, 1, 1)

        rel_pose_map = self.__rel_position(pose_embed)
        h_t = torch.cat([patches, tokens], 1)   # (n, 2*h_num*w_num, c)
        h_t = h_t.transpose(1,0) # batch first to length first
        for attn, fc in zip(self.attn_layers, self.fc_layers):
            h_t = fc(attn(h_t, rel_pose_map)) + h_t
        h_t = h_t.transpose(1,0) # length first to batch first
        
        h_t = h_t[:, w*h:].reshape(b, h, w, c).permute(0,3,1,2)
        h_t = self.P_deconv(h_t)
        return h_t