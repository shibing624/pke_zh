# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

"""

这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。答谢宴于晚上8点开始。

sentences:
这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足
答谢宴于晚上8点开始

words_no_filter
这/间/酒店/位于/北京/东三环/里面/摆放/很多/雕塑/文艺/气息/十足
答谢/宴于/晚上/8/点/开始

words_no_stop_words
间/酒店/位于/北京/东三环/里面/摆放/很多/雕塑/文艺/气息/十足
答谢/宴于/晚上/8/点

words_all_filters
酒店/位于/北京/东三环/摆放/雕塑/文艺/气息
答谢/宴于/晚上

"""

import sys

sys.path.append('..')
from wordrank.utils.segment import Segmentation


def main():
    text = "这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。答谢宴于晚上8点开始。"
    s = Segmentation()
    ret = s.segment(text)
    print(ret)


if __name__ == '__main__':
    main()
