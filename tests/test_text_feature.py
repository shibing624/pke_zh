# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import os
import unittest

sys.path.append('..')
from wordrank import TextFeature


class TextTestCase(unittest.TestCase):
    def test_get_feature(self):
        q = "哪里下载电视剧潜伏"
        t = TextFeature()
        a = t.get_feature(q)
        print(a)
        self.assertTrue(len(a) > 0)


if __name__ == '__main__':
    unittest.main()
