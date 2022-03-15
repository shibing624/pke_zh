# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest

sys.path.append('..')
from wordrank import WordRank

from wordrank.features.textrank import TextRank4Keyword
from wordrank.features.tfidf import TFIDF4Keyword

pwd_path = os.path.abspath(os.path.dirname(__file__))


class TextRankTestCase(unittest.TestCase):
    def test_textrank(self):
        m = WordRank()
        s = ['哪里下载电视剧潜伏',
             '一架飞机要起飞了',
             '一个男人在吹一支大笛子。',
             '一个人正把切碎的奶酪撒在比萨饼上。']
        for q in s:
            for x, w in TextRank4Keyword().extract_tags(q, withWeight=True):
                print('%s %s' % (x, w))
            print()
        print('-' * 42)
        for q in s:
            for x, w in TFIDF4Keyword().extract_tags(q, withWeight=True):
                print('%s %s' % (x, w))
            print()


if __name__ == '__main__':
    unittest.main()
