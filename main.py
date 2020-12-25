###################################################################
###   @FilePath: \GarNet\main.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 14:14:25
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-25 11:38:11
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
# -*- coding: utf-8 -*-

import os
import  argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'] )
parser.add_argument('--version', type=str, default='ViT1', help='experiment indicator')
parser.add_argument('--script', type=str, default='resnet', help='experiment indicator')
parser.add_argument('--description', type=str, default='training ViT on ImageNet2012', help='experiment indicator')
# params for model
parser.add_argument('--dim', type=int, default=256, help='tensor dimensions')
parser.add_argument('--depth', type=int, default=4, help='number of transformer encoder layers')
# params for train
parser.add_argument('--train_folder', type=str, default='../dataset/train_data', help='the training data path')
parser.add_argument('--log_dir', type=str, default='./logs', help='the testing label path')
parser.add_argument('--model_name', type=str, default='ViT', help='model used for training, cap sensitive')
parser.add_argument('--num_classes', type=int, default=43, help='the num of classes which your task should classify')
parser.add_argument('--batch_size', type=int, default=4, help='the num of classes which your task should classify')
parser.add_argument('--image_size', type=int, default=256, help='the size to which images are padded')
parser.add_argument('--patch_size', type=int, default=2, help='the size to which images are padded')
parser.add_argument('--max_int', type=int, default=16, help='the size to which images are padded')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
parser.add_argument('--max_epoch', type=int, default=20, help='')
parser.add_argument('--checkpoint', type=int, default=2000, help='')
parser.add_argument('--cuda_id', type=int, default=0, help='used cuda device')
config = parser.parse_args()


def main(argv=None):
    package  = __import__('scripts.train_'+config.script, fromlist=True)
    Trainer  = getattr(package, 'Trainer')

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
