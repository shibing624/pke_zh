# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

* 语言模型特征：整个query的语言模型概率/去掉该Term后的Query的语言模型概率。
"""

from copy import deepcopy

import pycorrector

from wordrank.features.text_feature import AttrDict
from wordrank.utils.tokenizer import word_segment


class LanguageFeature(object):
    def __init__(self, segment_sep=' '):
        self.segment_sep = segment_sep

    def get_ngram_score(self, words):
        """
        取n元文法得分
        :param words: list, 以词或字切分
        :param mode:
        :return:
        """
        return pycorrector.ngram_score(' '.join(words))

    def get_ppl_score(self, words):
        """
        取语言模型困惑度得分，越小句子越通顺
        :param words: list, 以词或字切分
        :param mode:
        :return:
        """
        return pycorrector.ppl_score(' '.join(words))

    def get_feature(self, query, is_word_segmented=False):
        """
        Get language feature
        :param query:
        :param is_word_segmented:
        :return: list, list: term features, sentence features
        """
        term_features = []
        if is_word_segmented:
            word_seq = query.split(self.segment_sep)
        else:
            word_seq = word_segment(query, cut_type='word', pos=False)

        # sentence
        sentence_features = AttrDict(
            ppl=self.get_ppl_score(word_seq),
        )

        # term
        count = 0
        for word in word_seq:
            word_list = deepcopy(word_seq)
            if word in word_list:
                word_list.remove(word)
            left_word = word_seq[count - 1] if count > 0 else ''
            right_word = word_seq[count + 1] if count < len(word_seq) - 1 else ''

            term_features.append(AttrDict(
                del_term_ppl=self.get_ppl_score(word_list),
                term_ngram_score=self.get_ngram_score(word),
                left_term_score=self.get_ngram_score(left_word + word),
                right_term_score=self.get_ngram_score(word + right_word)
            ))
            count += 1
        return term_features, sentence_features
