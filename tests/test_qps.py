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
from pke_zh.supervised.wordrank import WordRank

text = """
钟声文章啥意思
邪御天娇免费阅读
狼人杀魔术师
红灯时可以掉头吗
陈鲁豫为何不能生育
全能麟少txt下载
全员加速中若风在哪一期
入什么什么出成语
元阳县地图全图
余罪第三季评价
从小就有抬头纹命运
仁者不忧/智者不惑/勇者不惧
什么什么信心的四字词
人间完整版132分钟
井冈山景点介绍
二十厘米床多大
乳腺囊肿需要治疗吗
乳腺发育不良
乙醇危害性
中国富豪有小耳朵吗
上海行蕴信息科技有限公司
一席什么什么成语
usb总线最大传输速率是
qq头像女生霸气高冷范
lol季前赛什么意思
"""


class QPSPredictTestCase(unittest.TestCase):
    def test_predict_speed(self):
        m = WordRank()
        sents = text.strip().split()
        sents = sents * 10
        t1 = time()
        for sent in sents:
            r = m.extract(sent)
        spend_time = time() - t1
        print('sente size:', len(sents))
        print('spend time:', spend_time, ' seconds')
        print('rank qps:', len(sents) / spend_time)


if __name__ == '__main__':
    unittest.main()
