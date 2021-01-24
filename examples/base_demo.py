# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append("..")

pwd_path = os.path.abspath(os.path.dirname(__file__))

def is_common(w):
    return w in ['a','b']
if __name__ == '__main__':
    q = '哪里下载电视剧潜伏'
    a = 'a'
    b = is_common(a)
    print(b)
    a = ['a','b']
    b = 'b'
    c = [b]