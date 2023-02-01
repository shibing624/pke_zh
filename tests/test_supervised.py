# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest

sys.path.append('..')
from pke_zh.supervised.wordrank import WordRank

sents = [
    "我爱北京天安门",
    '哪里下载电视剧潜伏',
    '一架飞机要起飞了',
    "天安门上太阳升，太阳升起，我们怎么去天安门",
    "洗烘一体得价格算亲民了，特意挑选了一件小孩的裤子洗，洗的很干净，声音很小，容量很大。超爱！！"
]


class TestCase(unittest.TestCase):
    def test_wordrank_case(self):
        m = WordRank()
        print('sente size:', len(sents))
        for sent in sents:
            r = m.extract(sent)
            print(sent, r)


if __name__ == '__main__':
    unittest.main()
