# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

前后词互信息PMI: 计算两个词语之间的关联程度，pmi越大越紧密
左右邻熵：左右熵值越大，说明该词的周边词越丰富，意味着词的自由程度越大，其成为一个独立的词的可能性也就越大。
"""

import os
import re
from collections import Counter

import numpy as np

from wordrank import config
from wordrank.utils.io_utils import save_json, load_json
from wordrank.utils.logger import logger


def ngram_freq(text, ngram=4):
    """
    Get Ngram freq dict
    :param text: input text, sentence
    :param ngram: N
    :return: word frequency dict
    """
    words = []
    for i in range(1, ngram + 1):
        words += [text[j:j + i] for j in range(len(text) - i + 1)]
    words_freq = dict(Counter(words))
    return words_freq


def generate_pmi_score(word_freq_dict, pmi_path):
    """
    Generate PMI score
    :param word_freq_dict: dict
    :return: dict(word, float), pmi score
    """
    result = dict()
    for word in word_freq_dict:
        if len(word) > 1:
            p_x_y = min([word_freq_dict.get(word[:i]) * word_freq_dict.get(word[i:]) for i in range(1, len(word))])
            score = p_x_y / word_freq_dict.get(word)
            result[word] = score
    save_json(result, pmi_path)
    return result


def entropy_score(char_list):
    """
    Get entropy score
    :param char_list: input char list
    :return: float, score
    """
    char_freq_dict = dict(Counter(char_list))
    char_size = len(char_list)
    entropy = -1 * sum([char_freq_dict.get(i) / char_size * np.log2(char_freq_dict.get(i) / char_size)
                        for i in char_freq_dict])
    return entropy


def generate_entropy_score(word_freq_dict, text, entropy_path):
    """
    Generate entropy score
    :param word_freq_dict: dict
    :param text: input text, document
    :param entropy_path: save entropy file path
    :return: dict
    """
    result = dict()
    for word in word_freq_dict:
        if len(word) == 1:
            continue
        # pass error pattern
        if '*' in word:
            continue
        try:
            left_right_char = re.findall('(.)%s(.)' % word, text)
            left_char = [i[0] for i in left_right_char]
            left_entropy = entropy_score(left_char)

            right_char = [i[1] for i in left_right_char]
            right_entropy = entropy_score(right_char)
            if left_entropy > 0 or right_entropy > 0:
                result[word] = [left_entropy, right_entropy]
        except Exception as e:
            logger.error('error word %s, %s' % (word, e))
    save_json(result, entropy_path)
    return result


class PMI(object):
    def __init__(self,
                 text='',
                 ngram=4,
                 pmi_path=config.pmi_path,
                 entropy_path=config.entropy_path
                 ):
        if os.path.exists(pmi_path) and os.path.exists(entropy_path):
            self.pmi_score_dict = load_json(pmi_path)
            self.entropy_score_dict = load_json(entropy_path)
        elif len(text.strip()) > 0:
            words_freq = ngram_freq(text, ngram)
            self.pmi_score_dict = generate_pmi_score(words_freq, pmi_path)
            self.entropy_score_dict = generate_entropy_score(words_freq, text, entropy_path)
        else:
            raise ValueError("text and saved file path must exist one")

    def pmi_score(self, word):
        """
        Get PMI score
        :param word:
        :return:
        """
        return self.pmi_score_dict.get(word, 0.0)

    def entropy_score(self, word):
        """
        Get entropy score
        :param word:
        :return:
        """
        return self.entropy_score_dict.get(word, [0.0, 0.0])
