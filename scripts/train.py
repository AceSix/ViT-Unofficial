# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \ViT-Unofficial\scripts\train.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 14:14:25
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-24 10:35:13
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
import datetime

from Tools.data_loader import data_iterator
from Tools.logger import Logger

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.train_iter = data_iterator(config.train_label, config.train_folder, config.image_size)
        self.test_iter = data_iterator(config.test_label, config.test_folder, config.image_size)

        self.save_dir = os.path.join(config.log_dir, config.version)
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d_%H:%M:%S')
        self.logger = Logger(os.path.join(self.save_dir, f'record_{timeStr}.txt'))
        self.logger.log_param(config)

    def build_model(self):
        config = self.config
        package  = __import__('models.'+config.model_name, fromlist=True)
        model_object  = getattr(package, config.model_name)

        self.model = model_object(config.dim, config.num_classes, config.depth, config.image_size, config.patch_size)
        
    def train(self):
        config = self.config
        with tf.device("/GPU:0"):
            self.build_model()

            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

            test_loss = tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

            # @tf.function(experimental_relax_shapes=True)
            @tf.function()
            def train_step(images, labels):
                with tf.GradientTape() as tape:
                    predictions = self.model(images)
                    loss = loss_object(labels, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                train_loss(loss)
                train_accuracy(labels, predictions)

            # @tf.function(experimental_relax_shapes=True)
            @tf.function()
            def test_step(images, labels):
                predictions = self.model(images)
                t_loss = loss_object(labels, predictions)

                test_loss(t_loss)
                test_accuracy(labels, predictions)
        
            self.logger.log_text("Start training...\n")
            for i in range(config.max_iters):

                images, labels = self.train_iter.get_batch(config.batch_size)
                train_step(images, labels)

                if i%config.checkpoint==0:
                    for j in range(200):
                        images, labels = self.test_iter.get()
                        test_step(images, labels)

                    self.model.save_weights(os.path.join(self.checkpoint_dir, f'iter-{i}'))
                    timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
                    template = '[{}]-[Iter]-[{}]-[Loss]-[{:.2f}]-[Accuracy]-[{:.2f}]-[Test Loss]-[{:.2f}]-[Test Accuracy]-[{:.2f}]'
                    print (template.format(timeStr, i,
                           train_loss.result().numpy(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))

                    self.logger.record(
                        Iter=i, Loss=train_loss.result(), Accuracy=train_accuracy.result()*100,
                        TestLoss=test_loss.result(), TestAccuracy=test_accuracy.result()*100
                    )

                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    test_loss.reset_states()
                    test_accuracy.reset_states()