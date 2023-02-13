# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from pke_zh.unsupervised.textrank import TextRank

m = TextRank()
r = m.extract_sentences(
    """较早进入中国市场的星巴克，是不少小资钟情的品牌。相比 在美国的平民形象，星巴克在中国就显得“高端”得多。用料并无差别的一杯中杯美式咖啡，在美国仅约合人民币12元，国内要卖21元，相当于贵了75%。  第一财经日报""")
print(r)
