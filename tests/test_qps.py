# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from time import time

sys.path.append('..')
from wordrank.model import data_reader
from wordrank import WordRank

pwd_path = os.path.abspath(os.path.dirname(__file__))
test_path = os.path.join(pwd_path, '../extra_data/train.csv')

sents, labels = data_reader(test_path)


class QPSPredictTestCase(unittest.TestCase):
    def test_predict_speed(self):
        m = WordRank()
        t1 = time()
        for sent in sents:
            r = m.rank(sent)
            # print(r)
        spend_time = time() - t1
        print('sente size:', len(sents))
        print('spend time:', spend_time, ' seconds')
        print('rank qps:', len(sents) / spend_time)


if __name__ == '__main__':
    unittest.main()
