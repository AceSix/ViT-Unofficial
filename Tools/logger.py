# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \template_tf\Tools\logger.py
###   @Author: Ziang Liu
###   @Date: 2020-12-23 16:29:20
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-23 16:49:04
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import datetime
import prettytable as pt

class Logger(object):
    def __init__(self, path):
        self.path = path

    def log_param(self, config):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
            logf.writelines("[%s]-[Config]\n"%(timeStr))
            for i,item in enumerate(config.__dict__.items()):
                text = "[%d]-[parameters] %s--%s\n"%(i,item[0],str(item[1]))
                logf.writelines(text)

    def log_text(self, text):
        with open(self.path,'a+') as logf:
            logf.writelines(text)

    def record(self, **args):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
            text = "[timeStr]"
            for key in args.keys():
                text += f"-[{key}]-[{args[key]}]"
            text += "\n"
            logf.writelines(text)