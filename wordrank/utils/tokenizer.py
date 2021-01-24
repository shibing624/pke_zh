# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 配置切词器
"""
import re

import jieba
from jieba import posseg

from wordrank import config

jieba.setLogLevel(log_level="ERROR")


def word_segment(sentence, cut_type='word', pos=False):
    """
    Word segmentation
    :param sentence:
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence)
    :param pos: enable POS
    :return: list
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


def sentence_segment(text, delimiters=config.sentence_delimiters, include_symbols=False):
    """
    Sentence segmentation
    :param text: query
    :param delimiters: set
    :param include_symbols: bool
    :return: list(word, idx)
    """
    result = []
    delimiters = set([item for item in delimiters])
    delimiters_str = '|'.join(delimiters)
    blocks = re.split(delimiters_str, text)
    start_idx = 0
    for blk in blocks:
        if not blk:
            continue
        result.append((blk, start_idx))
        start_idx += len(blk)
        if include_symbols and start_idx < len(text):
            result.append((text[start_idx], start_idx))
            start_idx += 1
    return result


if __name__ == '__main__':
    text = "这个消息在北京城里不胫而走。你听过了吗？"

    words = '我，来。上海？吃？上海菜'
    wordlist = re.split('？|。', words)
    print(wordlist)

    print(text)
    t = word_segment(text, pos=True)
    print(t)
    t = sentence_segment(text, include_symbols=True)
    print(t)
    t = sentence_segment('这个消息在北京城', include_symbols=True)
    print(t)
