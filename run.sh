###
 # @FilePath: \ViT-Unofficial\run.sh
 # @Author: Ziang Liu
 # @Date: 2020-12-23 17:03:30
 # @LastEditors: Ziang Liu
 # @LastEditTime: 2020-12-24 10:01:44
 # @Copyright (C) 2020 SJTU. All rights reserved.
### 

python main.py --version "ViT2"\
               --description "imagenet pretrain" \
               --image_size 384 \
               --patch_size 12 \
               --dim 256 \
               --depth 4 \
               --train_folder /home/gdp/harddisk/Data1/Imagenet2012/train \
               --test_folder /home/gdp/harddisk/Data1/Imagenet2012/val \
               --train_label ./data/train_label.txt \
               --test_label ./data/validation_label.txt \
               --cuda_id 0