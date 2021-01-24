# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from wordrank.features.pmi import PMI
from wordrank import config

q = "天龙八部哪里看"

if __name__ == '__main__':
    # read the data and preprocessing the data to a whole str
    stop_word = ['【', '】', ')', '(', '、', '，', '“', '”', '。', '\n', '《', '》', ' ', '-', '！', '？', '.', '\'', '[', ']',
                 '：', '/', '.', '"', '\u3000', '’', '．', ',', '…', '?']
    text = ''
    count = 0
    with open(config.domain_sample_path) as f:
        for line in f:
            count += 1
            if count > 300:
                continue
            text += line.strip()
    for i in stop_word:
        text = text.replace(i, "")

    pmi_model = PMI(text,
                    ngram=4,
                    pmi_path=config.pmi_path,
                    entropy_path=config.entropy_path)
    print(pmi_model.pmi_score('天龙'))
    print(pmi_model.pmi_score('天龙八部'))
    print(pmi_model.pmi_score('龙八部哪'))

    print(pmi_model.entropy_score('天龙'))
    print(pmi_model.entropy_score('天龙八部'))
    print(pmi_model.entropy_score('龙八部哪'))
