# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \GarNet\Tools\data_loader.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 15:30:52
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-27 17:22:33
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
import torch
import torch.utils.data as data

from PIL import Image
from Tools.utils import train_trans, test_trans
import joblib

def load_image(path, trans):
    image = Image.open(path)
    # w,h = image.size
    # wh = max(w,h)
    # image = transforms.Pad((0,0,wh-w, wh-h), fill=0, padding_mode='constant')(image)
    return trans(image)

class Garbage(data.Dataset):
    def __init__(self, data, path, trans):
        self.dataset = data
        self.path = path
        self.trans = trans

    def __getitem__(self, index):
        name, label = self.dataset[index]
        image = load_image(os.path.join(self.path, name), self.trans)
        label = torch.LongTensor([int(label)])
        return image, label

    def __len__(self):
            return len(self.dataset)

def loadNsplit(path, image_size):
    train = joblib.load( 'train.pkl')
    test = joblib.load( 'test.pkl')
    train_text = train + test[:len(test)//2]
    test_text = test[len(test)//2:]
    print(f"A total of {len(train_text)+len(test_text)//2} training samples\nA total of {len(test_text)//2} testing samples")
    train_set = Garbage(train_text, path, train_trans(image_size))
    test_set = Garbage(test_text, path, test_trans(image_size))
    return train_set, test_set
