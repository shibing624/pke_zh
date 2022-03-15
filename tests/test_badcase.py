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
from wordrank import TextRank4Keyword, TFIDF4Keyword

pwd_path = os.path.abspath(os.path.dirname(__file__))
sents = ["我爱北京天安门",
         "天安门上太阳升，太阳升起，我们怎么去天安门",
         "洗烘一体得价格算亲民了，特意挑选了一件小孩的裤子洗，洗的很干净，声音很小，容量很大。超爱！！"]
m = WordRank()


class CaseTestCase(unittest.TestCase):
    def test_case(self):
        for sent in sents:
            r = m.rank(sent)
            print(r)
        print('sente size:', len(sents))

    def test_textrank(self):
        for s in sents:
            m = TextRank4Keyword()
            r = m.extract_tags(s)
            print(r)

        for s in sents:
            m = TextRank4Keyword()
            r = m.extract_tags(s, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False)
            print(r)

        for s in sents:
            m = TextRank4Keyword()
            r = m.extract_tags(s, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=True)
            print(r)

    def test_tfidf(self):
        for s in sents:
            m = TFIDF4Keyword()
            r = m.extract_tags(s)
            print(r)

        for s in sents:
            m = TFIDF4Keyword()
            r = m.extract_tags(s, withWeight=True, withFlag=False)
            print(r)

        for s in sents:
            m = TFIDF4Keyword()
            r = m.extract_tags(s, withWeight=False, withFlag=True)
            print(r)


if __name__ == '__main__':
    unittest.main()
