# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append("..")

from wordrank.features.textrank import TextRank
from wordrank.features.tfidf import TFIDF

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    q = '哪里下载电视剧潜伏'
    m = TextRank()
    for x, w in m.textrank(q, withWeight=True):
        print('%s %s' % (x, w))
    print()
    m = TFIDF()
    for x, w in m.extract_tags(q, withWeight=True):
        print('%s %s' % (x, w))
