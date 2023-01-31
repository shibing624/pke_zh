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
from pke_zh.unsupervised.textrank import TextRank
from pke_zh.unsupervised.tfidf import TfIdf

pwd_path = os.path.abspath(os.path.dirname(__file__))
sents = ["我爱北京天安门",
         "天安门上太阳升，太阳升起，我们怎么去天安门",
         "洗烘一体得价格算亲民了，特意挑选了一件小孩的裤子洗，洗的很干净，声音很小，容量很大。超爱！！"]


class CaseTestCase(unittest.TestCase):
    def test_wordrank_case(self):
        m = WordRank()
        print('sente size:', len(sents))
        for sent in sents:
            r = m.extract(sent)
            print(sent, r)

    def test_textrank(self):
        m = TextRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_tfidf(self):
        m = TfIdf()
        for s in sents:
            r = m.extract(s)
            print(s, r)


if __name__ == '__main__':
    unittest.main()
