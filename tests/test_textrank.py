# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest

sys.path.append('..')
from pke_zh.unsupervised.textrank import TextRank

pwd_path = os.path.abspath(os.path.dirname(__file__))
sents = [
    "天安门上太阳升，太阳升起，我们去天安门升国旗啦",
    "洗烘一体得价格算亲民了，特意挑选了一件小孩的裤子洗，洗的很干净，声音很小，容量很大。超爱！！"
]


class TextRankTestCase(unittest.TestCase):
    def test_textrank(self):
        m = TextRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)


if __name__ == '__main__':
    unittest.main()
