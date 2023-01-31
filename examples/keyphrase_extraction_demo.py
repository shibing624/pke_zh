# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from pke_zh.supervised.wordrank import WordRank

m = WordRank()

print(m.extract("哪里下载电视剧周恩来？"))
