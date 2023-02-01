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
from pke_zh.unsupervised.tfidf import TfIdf
from pke_zh.unsupervised.singlerank import SingleRank
from pke_zh.unsupervised.positionrank import PositionRank
from pke_zh.unsupervised.topicrank import TopicRank
from pke_zh.unsupervised.multipartiterank import MultipartiteRank
from pke_zh.unsupervised.yake import Yake
from pke_zh.unsupervised.keybert import KeyBert

pwd_path = os.path.abspath(os.path.dirname(__file__))
sents = [
    "我爱北京天安门",
    '哪里下载电视剧潜伏',
    '一架飞机要起飞了',
    "天安门上太阳升，太阳升起，我们怎么去天安门",
    "洗烘一体得价格算亲民了，特意挑选了一件小孩的裤子洗，洗的很干净，声音很小，容量很大。超爱！！"
]


class TestCase(unittest.TestCase):
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

    def test_SingleRank(self):
        m = SingleRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_PositionRank(self):
        m = PositionRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_TopicRank(self):
        m = TopicRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_MultipartiteRank(self):
        m = MultipartiteRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_Yake(self):
        m = Yake()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_KeyBert(self):
        m = KeyBert()
        for s in sents:
            r = m.extract(s)
            print(s, r)


if __name__ == '__main__':
    unittest.main()
