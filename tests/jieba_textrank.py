# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append("..")

from wordrank.features.textrank import TextRank4Keyword
from wordrank.features.tfidf import TFIDF4Keyword

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    s = ['哪里下载电视剧潜伏',
         '一架飞机要起飞了',
         '一个男人在吹一支大笛子。',
         '一个人正把切碎的奶酪撒在比萨饼上。']
    for q in s:
        for x, w in TextRank4Keyword().textrank(q, withWeight=True):
            print('%s %s' % (x, w))
        print()
    print('-'*42)
    for q in s:
        for x, w in TFIDF4Keyword().extract_tags(q, withWeight=True):
            print('%s %s' % (x, w))
        print()