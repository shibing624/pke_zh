# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
import wordrank
from wordrank import set_log_level
set_log_level("INFO")

if __name__ == '__main__':
    p = '哪里下载电视剧周恩来？qq头像女生霸气高冷范.。这是句号吧。井冈山景点介绍，大家一起来看看'
    print(wordrank.rank(p))
