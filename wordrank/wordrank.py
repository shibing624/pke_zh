# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from wordrank import config
from wordrank.features.language_feature import LanguageFeature
from wordrank.features.statistics_feature import StatisticsFeature
from wordrank.features.text_feature import TextFeature
from wordrank.utils import tokenizer
from wordrank.utils.text_utils import convert_to_unicode
from wordrank.utils.logger import logger


class WordRank:
    def __init__(self,
                 stopwords_path=config.stopwords_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 common_char_path=config.common_char_path,
                 domain_sample_path=config.domain_sample_path,
                 ngram=4,
                 pmi_path=config.pmi_path,
                 entropy_path=config.entropy_path,
                 sentence_delimiters = config.sentence_delimiters,
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
        self.inited = True

    def check_inited(self):
        if not self.inited:
            self._init()

    def rank_query(self, query):
        self.check_inited()
        text_terms, text_sents = self.text_feature.get_feature(query)
        stat_terms, stat_sents = self.statistics_feature.get_feature(query)
        lang_terms, lang_sents = self.language_feature.get_feature(query)
        terms = text_terms + stat_terms + lang_terms
        sents = text_sents + stat_sents + lang_sents
        logger.debug('term features: %s' % terms)
        logger.debug('sentence features: %s' % sents)


    def rank(self, text):
        """
        Word Rank
        :param text:
        :return:
        """
        self.check_inited()
        # unicode
        text = convert_to_unicode(text)
        # split to short sentence
        sentences = tokenizer.sentence_segment(text, delimiters=self.sentence_delimiters, include_symbols=True)
        for sentence, idx in sentences:
            pass
