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
    "吉利博越PRO在7月3日全新极客智能生态系统GKUI19发布会上正式亮相！受到广泛关注。",
    "洗烘一体得价格算亲民了，特意挑选了一件小孩的裤子洗，洗的很干净，声音很小，容量很大。超爱！！",
    "较早进入中国市场的星巴克，是不少小资钟情的品牌。相比在美国的平民形象，星巴克在中国就显得“高端”得多。用料并无差别的一杯中杯美式咖啡，在美国仅约合人民币12元，国内要卖21元，相当于贵了75%。 第一财经日报",
]


class TextRankTestCase(unittest.TestCase):
    def test_textrank_keywords(self):
        m = TextRank()
        for s in sents:
            r = m.extract(s)
            print(s, r)

    def test_textrank_sentences(self):
        m = TextRank()
        for s in sents:
            r = m.extract_sentences(s)
            print(s, r)


if __name__ == '__main__':
    unittest.main()
