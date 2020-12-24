###
 # @FilePath: \ViT\run.sh
 # @Author: Ziang Liu
 # @Date: 2020-12-23 17:03:30
 # @LastEditors: Ziang Liu
 # @LastEditTime: 2020-12-23 17:14:07
 # @Copyright (C) 2020 SJTU. All rights reserved.
### 

python main.py --version ViT1\
               --description "imagenet pretrain" \
               --dim 256 \
               --depth 4 \
               --train_folder /home/gdp/harddisk/Data1/Imagenet2012/train \
               --test_folder /home/gdp/harddisk/Data1/Imagenet2012/val \
               --train_label ./data/train_label.txt \
               --test_label ./data/validation_label.txt \
               --cuda_id 0