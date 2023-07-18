# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from glob import glob
import sys
import os

sys.path.append('..')
from pke_zh.wordrank import PMI

pwd_path = os.path.abspath(os.path.dirname(__file__))

text = """
        不难看出，CG并不考虑在搜索结果的排序信息，CG得分高只能说明这个结果页面总体的质量比较高并不能说明这个算法做的排序好或差。
        在上面例子中，如果调换第二个结果和第三个结果的位置CG=5+2+3+1+2=13，并没有改变总体的得分。
        国际大蒜贸易网价格频道为您提供全国最新大蒜价格、蒜苔价格、蒜米价格、蒜片价格、洋葱价格、大蒜出口代加工价格、大蒜批发价格、大蒜冷库租赁价格等信息与服务
        综合几个大蒜主产区产量数据分析，今年大蒜产量减产率为8--10%左右。 今天是6月1日，一般杂交混级蒜价
        格为4.60—5.00元/公斤，去年同期一般混级蒜价格为2.00--2.20元/公斤，同比上涨率为128.57%，去年新蒜上市后到
        6月1日价格是一路下滑，今年新蒜上市后到6月1日价格走势基本上是一路上涨。
        小说天龙八部上市后，很多小说都流行起来。
"""


def read_file(file_path):
    D = open(file_path, 'r', encoding='utf-8').read()
    return D


if __name__ == '__main__':
    file_dir = os.path.join(pwd_path, 'data')
    files = glob(os.path.join(file_dir, '*.txt'), recursive=True)
    for i in files:
        print(i)
        text += read_file(i)
    print(len(text))
    text = text[:600000]
    print(len(text))

    query = ['大蒜', '小说', '价格']
    a = PMI(text, ngram=4, pmi_path='pmi_word_score.json', entropy_path='entropy_word_score.json')
    for i in query:
        r = a.pmi_score(i)
        print(f"pmi_score {i}: {r}")
        e = a.entropy_score(i)
        print(f"entropy_score {i}: {e}")
