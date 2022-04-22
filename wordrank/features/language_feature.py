# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

* 语言模型特征：整个query的语言模型概率/去掉该Term后的Query的语言模型概率。
"""

from copy import deepcopy

import os

from wordrank.features.text_feature import AttrDict
from wordrank.utils.tokenizer import word_segment
from wordrank.utils.logger import logger
from wordrank.utils.file_utils import get_file


class NGram:
    def __init__(self, model_name_or_path=None, cache_folder=os.path.expanduser('~/.pycorrector/datasets/')):
        if model_name_or_path and os.path.exists(model_name_or_path):
            logger.info('Load kenlm language model:{}'.format(model_name_or_path))
            language_model_path = model_name_or_path
        else:
            # 语言模型 2.95GB
            get_file(
                'zh_giga.no_cna_cmn.prune01244.klm',
                'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
                extract=True,
                cache_subdir=cache_folder,
                verbose=1)
            language_model_path = os.path.join(cache_folder, 'zh_giga.no_cna_cmn.prune01244.klm')
        try:
            import kenlm
        except ImportError:
            raise ImportError('Kenlm not installed, use "pip install kenlm".')
        self.lm = kenlm.Model(language_model_path)
        logger.debug('Loaded language model: %s.' % language_model_path)

    def ngram_score(self, sentence: str):
        """
        取n元文法得分
        :param sentence: str, 输入的句子
        :return:
        """
        return self.lm.score(' '.join(sentence), bos=False, eos=False)

    def perplexity(self, sentence: str):
        """
        取语言模型困惑度得分，越小句子越通顺
        :param sentence: str, 输入的句子
        :return:
        """
        return self.lm.perplexity(' '.join(sentence))


class LanguageFeature(object):
    def __init__(self, segment_sep=' '):
        self.segment_sep = segment_sep
        self.ngram = NGram()

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
            ppl=self.ngram.perplexity(word_seq),
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
                del_term_ppl=self.ngram.perplexity(word_list),
                term_ngram_score=self.ngram.ngram_score(word),
                left_term_score=self.ngram.ngram_score(left_word + word),
                right_term_score=self.ngram.ngram_score(word + right_word)
            ))
            count += 1
        return term_features, sentence_features
