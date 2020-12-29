# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\Tools\utils.py
###   @Author: Ziang Liu
###   @Date: 2020-12-24 19:23:03
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-29 09:55:01
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
import shutil
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def code_transfer(source, target, names):
    for name in names:
        shutil.copyfile(os.path.join(source, name), 
                        os.path.join(target, name))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def train_trans(image_size):
    trans = transforms.Compose([
                transforms.RandomAffine(15, translate=(0.1,0.1)),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(0.1,0.1,0.1),
                transforms.RandomPerspective(),
                # transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(),
            ])
    return trans

def test_trans(image_size):
    trans = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.ToTensor(),
                normalize,
            ])
    return trans

def plot_loss_curve(path, losses):
    fig = plt.figure()
    x = range(len(losses['train']))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, losses['train'], label='train')
    ax1.set_ylabel('train loss')
    ax1.set_title("learning curve")
    ax2 = ax1.twinx() 
    ax2.plot(x, losses['test'], 'r', label='test')
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel('test loss')
    ax2.set_xlabel('iteration')
    plt.legend()
    plt.savefig(path)
    plt.close(fig) 
    plt.clf()

class Precision(object):
    def __init__(self):
        self.true_num = 0
        self.all = 0

    def add(self, pre_tensor, trg_tensor):
        pred = pre_tensor.detach().cpu().numpy().argmax(-1)
        trg = trg_tensor.squeeze().detach().cpu().numpy()
        right_num = (pred==trg).sum()
        self.true_num += right_num
        self.all += pre_tensor.shape[0]

    def result(self):
        return float(self.true_num)/float(self.all)
    
    def reset(self):
        self.true_num = 0
        self.all = 0

class Matric(object):
    def __init__(self):
        self.holder = 0
        self.count = 0

    def add(self, loss):
        self.holder += loss.detach().cpu().numpy()
        self.count += 1

    def result(self):
        return float(self.holder)/float(self.count)
    
    def reset(self):
        self.holder = 0
        self.count = 0