# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\scripts\train_transfer.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 14:14:25
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-29 17:20:27
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
import os
import datetime

import torch
from torchvision.utils import save_image

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
        code_transfer("./Tools", self.code_dir, ['data_loader.py', 'logger.py', 'utils.py'])

        self.config = config

    def build_model(self):
        config = self.config
        if config.model_name=='ViT':
            code_transfer("./models", self.code_dir, ['ViT_google.py'])
            from models.ViT_google import ViT_google
            self.model = ViT_google(image_size=config.image_size, num_classes=config.num_classes).cuda()
        elif config.model_name=='ResNeXt':
            code_transfer("./models", self.code_dir, ['ResNeXt.py'])
            from models.ResNeXt import ResNeXt_pretrained
            self.model = ResNeXt_pretrained(nlabels=config.num_classes).cuda()
        elif config.model_name=='ViTResNeXt':
            code_transfer("./models", self.code_dir, ['ViT.py'])
            from models.ViT import ViT_ResNeXt
            self.model = ViT_ResNeXt(cls_num=config.num_classes, depth=config.depth, version=config.resnet).cuda()
        
    def train(self):
        self.build_model()
        config = self.config
        losses = {'train':[], 'test':[]}
        optimizer = torch.optim.Adam(
            [{'params': self.model.backbone.parameters(), 'lr': self.config.learning_rate/2.},
             {'params': self.model.classifier.parameters()}], 
            lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch)
        criterion = torch.nn.CrossEntropyLoss().cuda()

        train_metric = Precision()
        loss_metric = Matric()

        iters = 0
        for i in range(config.max_epoch):
            data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True)
            for _,(image, label) in enumerate(data_loader):
                image, label = image.cuda(), label[:,0].cuda()
    
                prediction = self.model(image)
                loss = criterion(prediction, label)*config.scale
    
                train_metric.add(prediction, label)
                loss_metric.add(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iters%config.checkpoint==0:
                    record = self.logger.record(Iter=iters, 
                                                Loss=loss_metric.result(), 
                                                TrainAcc=train_metric.result()*100,
                                                TestAcc=self.test()*100)
                    print(record)
                    if iters%(config.checkpoint*10):
                        torch.save(self.model.state_dict(), 
                                   os.path.join(self.checkpoint_dir, f'iter_{iters}.pth'))

                    plot_loss_curve(os.path.join(self.save_dir, 'loss.jpg'), losses)
                    train_metric.reset()
                    loss_metric.reset()
                iters+=1
            
            scheduler.step()
    
    def test(self):
        test_metric = Precision()
        self.model.eval()

        data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.config.test_batch_size, shuffle=True)
        with torch.no_grad():
            for j,(image, label) in enumerate(data_loader):
                image, label = image.cuda(), label.cuda()
                prediction = self.model(image)
                test_metric.add(prediction, label)
                
        self.model.train()
        return test_metric.result()

    def warmup_lr(self):
        lrs = []
        tmp = self.config.learning_rate
        for j in range(self.config.max_epoch):
            tmp = tmp*0.95
            lrs.append(tmp)
        return lrs

        