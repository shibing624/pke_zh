# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from wordrank.features import text_feature

q = "哪里下载电视剧潜伏"


def main():
    t = text_feature.TextFeature()
    a = t.get_feature(q)
    print(a)


if __name__ == '__main__':
    main()
