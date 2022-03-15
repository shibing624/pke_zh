# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append("..")
from wordrank import TFIDF4Keyword, TextRank4Keyword

if __name__ == '__main__':
    q = "物流很快，服务也很好，还有售后回馈。外观很时尚并且超大视野"
    r = TFIDF4Keyword().extract_tags(q)
    print(r)

    r = TextRank4Keyword().extract_tags(q)
    print(r)

    q = "哪里下载电视剧周恩来？"
    r = TFIDF4Keyword().extract_tags(q)
    print(r)

    r = TextRank4Keyword().extract_tags(q)
    print(r)