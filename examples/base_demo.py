# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from wordrank import WordRank

m = WordRank()

print(m.rank("哪里下载电视剧周恩来？"))
