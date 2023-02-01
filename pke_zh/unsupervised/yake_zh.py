# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Seon
@description:

YAKE(Yet Another Keyword Extractor)，一种基于关键词统计的单文档无监督关键词提取算法：
基于5种指标：是否大写，词的位置，词频，上下文关系，词在句中频率，来计算候选词的得分，从而筛选Top-N关键词。
中文只用后4个指标。

modify from: https://pypi.org/project/iyake-cn/0.5.5/#files
"""
import os
import jieba
import pandas as pd
import re
from math import log10, sqrt
from jieba import lcut, posseg

pwd_path = os.path.abspath(os.path.dirname(__file__))

# inner data file
default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')
jieba.setLogLevel('ERROR')


def get_pos_lst(words_lst):
    # 分词的位置列表
    return list(zip(words_lst, range(1, len(words_lst) + 1)))


def get_T_pos(pos_lst, word, median_fn):
    # 单个词的T_pos指标
    _lst = [i[1] for i in pos_lst if word in i[0]]
    _lst.sort()
    half = len(_lst) // 2
    median = (_lst[half] + _lst[~half]) / 2
    if median_fn is None:
        return log10(log10(median + 10)) + 1
    if callable(median_fn):
        return median_fn(median)


def words_count(words_lst):
    # 全文词频统计，返回字典
    counts_dict = {}
    for word in words_lst:
        counts_dict[word] = counts_dict.get(word, 0) + 1
    return counts_dict


def related_content(words_lst, word, size):
    # 单个词的T_Rel指标
    split_lst = ' '.join(words_lst).split(word)
    left_lst = split_lst[:-1]
    right_lst = split_lst[1:]
    DL = 0
    for i in left_lst:
        # 从 word 出现的每个地方往左取 size 个词，计算不重复词数
        left_words = i.split(' ')[-2: -2 - size: -1]
        DL += len(set(left_words))

    DR = 0
    for i in right_lst:
        # 从 word 出现的每个地方往右
        right_words = i.split(' ')[1: size + 1]
        DR += len(set(right_words))

    return DL / len(left_lst) + DR / len(right_lst)


def get_pseg(x):
    # 词性
    return [p for w, p in list(posseg.cut(x))][0]


def std(lst):
    ex = float(sum(lst)) / len(lst)
    s = 0
    for i in lst:
        s += (i - ex) ** 2
    return sqrt(float(s) / len(lst))


def get_S_t(content, only_cn=False, stop=None, pos_type='s', median_fn=None, tf_normal='yake', adjust=1,
            r_size=10):
    # 各项指标原始得分表
    if only_cn:
        # 纯中文分词
        clean_str = re.sub(r'[^一-龟，；。？！（）“”]+', '', content)
    else:
        clean_str = re.sub(r'[0-9]+', '', content)

    jb_lst = [w for w in lcut(clean_str) if len(w) > 1]  # 分词过滤单字

    if stop is not None:
        # 停用词
        jb_lst = [w for w in jb_lst if w not in stop]

    uni_lst = list(set(jb_lst))
    uni_lst.sort(key=jb_lst.index)  # 固定顺序唯一词
    split_content = re.split(r'[，；。？！]', clean_str)  # 原文本按标点拆分句子列表

    # 位置 T_pos 得分表
    pos_lst = 0
    if pos_type == 'w':
        pos_lst = get_pos_lst(jb_lst)  # 计算全文词位置
    elif pos_type == 's':
        pos_lst = get_pos_lst(split_content)  # 计算含词的句子位置
    t_pos_scores = []
    for w in uni_lst:
        t_pos_scores.append((w, get_T_pos(pos_lst, w, median_fn)))

    # 全文词频 TF_norm 得分表
    wc_dic = words_count(jb_lst)
    mean_tf = sum(wc_dic.values()) / len(uni_lst)  # 词频均值
    std_tf = std(list(wc_dic.values()))  # 词频标准差
    max_tf = max(list(wc_dic.values()))
    min_tf = min(list(wc_dic.values()))
    if max_tf - min_tf == 0:
        tf_normal = 'yake'
    tf_norm_scores = []
    TF_norm = 0
    for w in uni_lst:
        if tf_normal == 'yake':
            TF_norm = wc_dic.get(w) / (mean_tf + std_tf)  # yake版归一化
        if tf_normal == 'mm':
            TF_norm = (wc_dic.get(w) - min_tf) / (max_tf - min_tf)  # max-min归一化
        tf_norm_scores.append((w, TF_norm))

    # 上下文 T_Rel 得分表
    t_rel_scores = []
    all_words = len(uni_lst)
    for w in uni_lst:
        DL_RL = related_content(jb_lst, w, r_size)
        T_Rel = 1 + DL_RL * wc_dic.get(w) / all_words
        t_rel_scores.append((w, T_Rel))

    # 句间词频 T_sentence 得分表
    len_content = len(split_content)
    SF_dic = {}
    for w in uni_lst:
        for sentence in split_content:
            if w in sentence:
                SF_dic[w] = SF_dic.get(w, 0) + 1
    t_sentence_scores = [(i[0], i[1] / len_content) for i in list(SF_dic.items())]

    # 重要性 S_t 总分表
    df_scores = pd.DataFrame({
        'word': uni_lst,
        'fre': [wc_dic.get(i) for i in uni_lst],
        't_pos': [i[1] for i in t_pos_scores],
        'tf_norm': [i[1] for i in tf_norm_scores],
        't_rel': [i[1] for i in t_rel_scores],
        't_sentence': [i[1] for i in t_sentence_scores]
    })
    df_scores['pseg'] = df_scores['word'].apply(get_pseg)
    df_scores.eval(f's_t = t_pos*t_rel / ({adjust} + (tf_norm + t_sentence)/t_rel)', inplace=True)
    return df_scores


def get_key_words(df_scores, n_best=10, sort_col='s_t', ascend=True, p=None):
    # 获取关键词列表，默认前10个，升序
    if p is None:
        p = {'n', 'a'}
    p = list(p)
    df_scores = df_scores[df_scores['pseg'].isin(p)]
    df_items = df_scores.sort_values(sort_col, ascending=ascend)
    print(df_items)
    word_scores = zip(df_items['word'].to_list(), df_items['s_t'].to_list())
    return list(word_scores)[:n_best]


def get_stopwords(txt_file):
    return set([line.strip() for line in open(txt_file, 'r', encoding='utf-8').readlines()])


class YakeZH:
    def __init__(self, stopwords_path=None):
        if stopwords_path is None:
            stopwords_path = default_stopwords_path
        self.stopwords = get_stopwords(stopwords_path)

    def extract(self, text, n_best=10):
        df = get_S_t(text, stop=self.stopwords)
        keyphrases = get_key_words(df, n_best=n_best)
        return keyphrases


if __name__ == '__main__':
    txt = '物流很快，服务也很好，还有售后回馈。外观很时尚并且超大视野'
    m = YakeZH()
    words = m.extract(txt)
    print(words)
