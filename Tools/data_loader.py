# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \ViT-Unofficial\Tools\data_loader.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 15:30:52
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-24 18:25:47
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
import random
import numpy as np
import tensorflow as tf


class data_iterator(object):
    def __init__(self, label_path, image_dir, pad_size=384):
        self.dataset = []
        with open(label_path) as f:
            lines = f.readlines()
        for line in lines:
            self.dataset.append(line.strip().split(' '))
        self.image_dir = image_dir
        self.pad_size = pad_size

        random.shuffle(self.dataset)
        self.data_iter = iter(self.dataset)
    
    def get(self):
        try:
            image_path, label = next(self.data_iter)
            path = os.path.join(self.image_dir, image_path)
            img_raw = tf.io.read_file(path)
            img_tensor = tf.image.decode_image(img_raw, dtype=tf.float32)
            img_tensor = tf.image.resize_with_pad(img_tensor, self.pad_size, self.pad_size)
            img_tensor = img_tensor/255.0
            if tf.shape(img_tensor)[-1]==1:
                img_tensor = tf.repeat(img_tensor, [3], axis=-1)
            if tf.shape(img_tensor)[-1]>3:
                img_tensor = img_tensor[...,:3]
            label = tf.convert_to_tensor([[int(label)]])
            return img_tensor[tf.newaxis, ...], label
        except:
            random.shuffle(self.dataset)
            self.data_iter = iter(self.dataset)
            image, label = self.get()
            return image, label

    def get_batch(self, batch_size):
        images, labels = [],[]
        for _ in range(batch_size):
            image, label = self.get()
            images.append(image)
            labels.append(label)
        image_batch = tf.concat(images, axis=0)
        image_batch = tf.image.per_image_standardization(image_batch)
        label_batch = tf.concat(labels, axis=0)
        return image_batch, label_batch