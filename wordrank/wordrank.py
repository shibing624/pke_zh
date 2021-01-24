# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import numpy as np

from wordrank import config
from wordrank.features.language_feature import LanguageFeature
from wordrank.features.statistics_feature import StatisticsFeature
from wordrank.features.text_feature import TextFeature
from wordrank.utils import tokenizer
from wordrank.utils.io_utils import load_pkl
from wordrank.utils.logger import logger
from wordrank.utils.text_utils import convert_to_unicode


class WordRank(object):
    def __init__(self,
                 stopwords_path=config.stopwords_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 common_char_path=config.common_char_path,
                 domain_sample_path=config.domain_sample_path,
                 ngram=4,
                 pmi_path=config.pmi_path,
                 entropy_path=config.entropy_path,
                 sentence_delimiters=config.sentence_delimiters,
                 model_path=config.model_path,
                 ):
        self.stopwords_path = stopwords_path
        self.person_name_path = person_name_path
        self.place_name_path = place_name_path
        self.common_char_path = common_char_path

        self.domain_sample_path = domain_sample_path
        self.ngram = ngram
        self.pmi_path = pmi_path
        self.entropy_path = entropy_path
        self.sentence_delimiters = sentence_delimiters
        self.model_path = model_path
        self.inited = False

    def _init(self):
        self.text_feature = TextFeature(
            stopwords_path=self.stopwords_path,
            person_name_path=self.person_name_path,
            place_name_path=self.place_name_path,
            common_char_path=self.common_char_path)
        self.statistics_feature = StatisticsFeature(
            domain_sample_path=self.domain_sample_path,
            ngram=self.ngram,
            pmi_path=self.pmi_path,
            entropy_path=self.entropy_path)
        self.language_feature = LanguageFeature()
        self.model = load_pkl(self.model_path)
        self.inited = True

    def check_inited(self):
        if not self.inited:
            self._init()

    def get_feature(self, query):
        features = []
        terms = []

        text_terms, text_sent = self.text_feature.get_feature(query, is_word_segmented=False)
        stat_terms, stat_sent = self.statistics_feature.get_feature(query, is_word_segmented=False)
        lang_terms, lang_sent = self.language_feature.get_feature(query, is_word_segmented=False)
        # sentence feature
        text_sent.update(stat_sent)
        text_sent.update(lang_sent)
        logger.debug('sentence features: %s' % text_sent)
        sent_feature = [text_sent.query_length, text_sent.term_size, text_sent.ppl]
        # term feature
        for text, stat, lang in zip(text_terms, stat_terms, lang_terms):
            text.update(stat)
            text.update(lang)
            logger.debug('term features: %s' % text)
            term_feature = [text.term_length, text.idx, text.offset, float(text.is_number),
                            float(text.is_chinese), float(text.is_alphabet), float(text.is_stopword),
                            float(text.is_name), float(text.is_common_char), text.embedding_sum, text.del_term_score,
                            text.idf, text.text_rank_score, text.tfidf_score, text.pmi_score, text.left_entropy_score,
                            text.right_entropy_score, text.del_term_ppl, text.term_ngram_score, text.left_term_score,
                            text.right_term_score]
            feature = sent_feature + term_feature
            features.append(feature)
            terms.append(text.term)
        logger.info("features size: %s" % len(features))
        data_feature = np.array(features, dtype=float)
        return data_feature, terms

    def rank_query(self, query):
        self.check_inited()
        # get feature
        data_feature, terms = self.get_feature(query)
        # predict model
        label_pred = self.model.predict(data_feature)
        logger.debug("predict label: %s" % label_pred)
        return zip(terms, label_pred)

    def rank(self, text):
        """
        Word Rank
        :param text:
        :return:
        """
        result = []
        self.check_inited()
        # unicode
        text = convert_to_unicode(text)
        # split to short sentence
        sentences = tokenizer.sentence_segment(text, delimiters=self.sentence_delimiters, include_symbols=True)
        for sentence, idx in sentences:
            for w, p in self.rank_query(sentence):
                result.append((w, p))
        return result
