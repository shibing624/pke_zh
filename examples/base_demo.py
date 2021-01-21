# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append("..")

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    q = '哪里下载电视剧潜伏'
    import jieba.analyse

    for x, w in jieba.analyse.textrank(q, withWeight=True, withFlag=True):
        print('%s %s' % (x, w))
