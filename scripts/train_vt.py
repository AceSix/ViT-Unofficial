# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\scripts\train_vt.py
###   @Author: Ziang Liu
###   @Date: 2020-12-24 19:05:28
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-25 16:15:09
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
import os
import datetime

import torch
from torchvision.utils import save_image

from models.VT import VT
from models.Blocks import ResNetBlock
from Tools.data_loader import loadNsplit
from Tools.logger import Logger
from Tools.utils import code_transfer, plot_loss_curve, Precision, Matric

class Trainer(object):
    def __init__(self, config):
        self.save_dir = os.path.join(config.log_dir, config.version)
        self.code_dir = os.path.join(self.save_dir, 'codes')
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d_%H-%M-%S')
        self.logger = Logger(os.path.join(self.save_dir, f'record_{timeStr}.txt'))
        self.logger.log_param(config)

        self.train_data, self.test_data = loadNsplit(config.train_folder, config.image_size)

        code_transfer("./", self.code_dir, ['run.sh'])
        code_transfer("./scripts", self.code_dir, [f'train_{config.script}.py'])
        code_transfer("./models", self.code_dir, ['ViT.py', 'ResNet.py'])
        code_transfer("./Tools", self.code_dir, ['data_loader.py', 'logger.py', 'utils.py'])

        self.resnet = ResNetBlock().cuda()
        self.model = VT(dim_hid=config.dim, dim_KQ=config.dim, cls_num=config.num_classes, 
                         patch=config.patch_size, stride=config.patch_size).cuda()
        self.config = config

        
    def train(self):
        config = self.config
        losses = {'train':[], 'test':[]}
        optimizer1 = torch.optim.Adam(self.resnet.parameters(), lr=config.learning_rate)
        optimizer2 = torch.optim.Adam(self.transformer.parameters(), lr=config.learning_rate/2.)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=200)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)
        criterion = torch.nn.CrossEntropyLoss()

        train_metric = Precision()
        loss_metric = Matric()

        iters = 0
        for i in range(config.max_epoch):
            data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True)
            for _,(image, label) in enumerate(data_loader):
                image, label = image.cuda(), label[:,0].cuda()
    
                prediction = self.transformer(self.resnet(image))
                loss = criterion(prediction, label)*128
    
                train_metric.add(prediction, label)
                loss_metric.add(loss)

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                if iters%config.checkpoint==0:
                    record = self.logger.record(Iter=iters, 
                                                Loss=loss_metric.result(), 
                                                TrainAcc=train_metric.result()*100,
                                                TestAcc=self.test()*100)
                    print(record)
                    if iters%(config.checkpoint*10):
                        torch.save([self.resnet.state_dict(), self.transformer.state_dict()], 
                                   os.path.join(self.checkpoint_dir, f'iter_{iters}.pth'))

                    plot_loss_curve(os.path.join(self.save_dir, 'loss.jpg'), losses)
                    train_metric.reset()
                    loss_metric.reset()

                iters+=1
            
            scheduler1.step()
            scheduler2.step()
    
    def test(self):
        test_metric = Precision()
        self.transformer.eval()
        self.resnet.eval()

        data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.config.batch_size, shuffle=True)
        with torch.no_grad():
            for j,(image, label) in enumerate(data_loader):
                image, label = image.cuda(), label.cuda()
                prediction = self.transformer(self.resnet(image))
                test_metric.add(prediction, label)
                if j*self.config.batch_size>500:
                    break
                
        self.transformer.train()
        self.resnet.train()
        return test_metric.result()
