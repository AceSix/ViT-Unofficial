# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \ViT-Unofficial\models\ViT.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 14:14:26
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-24 09:58:48
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Attention
from tensorflow.keras import Model

class PositionEmbedding(Model):
    def __init__(self, dim, max_int=512):
        super(PositionEmbedding, self).__init__()
        self.emb_x = Embedding(max_int, dim)
        self.emb_y = Embedding(max_int, dim)

    def call(self, patches):
        b,w,h = tf.shape(patches)[0],tf.shape(patches)[1],tf.shape(patches)[2]
        pose_x = tf.range(w, delta=1)[tf.newaxis, ...]
        pose_y = tf.range(h, delta=1)[tf.newaxis, ...]
        pose_x = tf.repeat(pose_x, [b], axis=0)
        pose_y = tf.repeat(pose_y, [b], axis=0)
        emb_x = self.emb_x(pose_x)[:,:,tf.newaxis,:]
        emb_y = self.emb_y(pose_y)[:,tf.newaxis,:,:]
        z = emb_x*emb_y
        z = tf.reshape(z, [tf.shape(z)[0],-1,tf.shape(z)[-1]])
        return z
    
class Encoder(Model):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.LN1 = LayerNormalization()
        self.LN2 = LayerNormalization()
        self.attention = Attention()
        self.embedding = Dense(dim*3, activation='relu')
        self.MLP = tf.keras.Sequential()
        self.MLP.add(Dense(dim, activation='relu'))
        self.MLP.add(Dense(dim, activation='relu'))
        self.dim = dim

    def call(self, x):
        dim = self.dim
        x_ = self.LN1(x)
        qvk = self.embedding(x_)
        q,v,k = qvk[...,:dim],qvk[...,dim:dim*2],qvk[...,dim*2:dim*3]
        y = self.attention([q,v,k]) + x
        y_ = self.LN2(y)
        y = self.MLP(y_) + y
        return y
    
class ViT(Model):
    def __init__(self, dim, fields, depth, image_size, patch_size):
        super(ViT, self).__init__()
        self.PE = PositionEmbedding(dim, max_int=image_size//patch_size+1)
        self.FC = Dense(dim, activation='relu')
        self.encoder = tf.keras.Sequential()
        for i in range(depth):
            self.encoder.add(Encoder(dim))
        self.cls = tf.Variable(tf.random.truncated_normal([1, 1, dim]), trainable=True)
        self.out = tf.keras.Sequential()
        self.out.add(Dense(dim, activation='relu'))
        self.out.add(Dense(fields, activation='softmax'))

        self.pc = patch_size

    def call(self, x):
        patches = tf.image.extract_patches(images=x,
                           sizes=[1, self.pc, self.pc, 1],
                           strides=[1, self.pc, self.pc, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
        pose_emb = self.PE(patches)
        
        patches = tf.reshape(patches, [tf.shape(patches)[0],-1,tf.shape(patches)[-1]])
        img_emb = self.FC(patches)
        
        embedding = img_emb + pose_emb
        embedding = tf.concat([tf.repeat(self.cls, [x.shape[0]], axis=0), embedding], 1)
        z = self.encoder(embedding)[:,0,:]
        z = self.out(z)
        return z