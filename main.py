###################################################################
###   @FilePath: \ViT-Unofficial\main.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 14:14:25
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-24 16:13:58
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
# -*- coding: utf-8 -*-

import os
import  argparse
import tensorflow as tf

from scripts.train import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'] )
parser.add_argument('--version', type=str, default='ViT1', help='experiment indicator')
parser.add_argument('--description', type=str, default='training ViT on ImageNet2012', help='experiment indicator')
# params for model
parser.add_argument('--dim', type=int, default=256, help='tensor dimensions')
parser.add_argument('--depth', type=int, default=4, help='number of transformer encoder layers')
# params for train
parser.add_argument('--train_folder', type=str, default='', help='the training data path')
parser.add_argument('--train_label', type=str, default='', help='the training label path')
parser.add_argument('--test_folder', type=str, default='', help='the testing data path')
parser.add_argument('--test_label', type=str, default='', help='the testing label path')
parser.add_argument('--log_dir', type=str, default='./logs', help='the testing label path')
parser.add_argument('--model_name', type=str, default='ViT', help='model used for training, cap sensitive')
parser.add_argument('--num_classes', type=int, default=1000, help='the num of classes which your task should classify')
parser.add_argument('--batch_size', type=int, default=4, help='the num of classes which your task should classify')
parser.add_argument('--image_size', type=int, default=512, help='the size to which images are padded')
parser.add_argument('--max_int', type=int, default=16, help='the size to which images are padded')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
parser.add_argument('--max_iters', type=int, default=2000000, help='')
parser.add_argument('--checkpoint', type=int, default=2000, help='')
parser.add_argument('--cuda_id', type=int, default=0, help='used cuda device')
config = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(config.cuda_id)

def main(argv=None):
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
