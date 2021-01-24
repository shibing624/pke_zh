# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
import wordrank

if __name__ == '__main__':
    q = '哪里下载电视剧周恩来'
    r = wordrank.rank(q)
    print(r)

    p = '哪里下载电视剧周恩来？'
    print(wordrank.rank(p))
