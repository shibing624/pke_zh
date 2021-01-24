# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from wordrank import config
from wordrank.features.language_feature import LanguageFeature
from wordrank.features.statistics_feature import StatisticsFeature
from wordrank.features.text_feature import TextFeature
from wordrank.utils.logger import logger


class Feature(object):
    def __init__(self,
                 stopwords_path=config.stopwords_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 common_char_path=config.common_char_path,
                 segment_sep=config.segment_sep,
                 domain_sample_path=config.domain_sample_path,
                 ngram=config.ngram,
                 pmi_path=config.pmi_path,
                 entropy_path=config.entropy_path,
                 sentence_delimiters=config.sentence_delimiters,
                 ):
        self.stopwords_path = stopwords_path
        self.person_name_path = person_name_path
        self.place_name_path = place_name_path
        self.common_char_path = common_char_path
        self.segment_sep = segment_sep

        self.domain_sample_path = domain_sample_path
        self.ngram = ngram
        self.pmi_path = pmi_path
        self.entropy_path = entropy_path
        self.sentence_delimiters = sentence_delimiters
        self.feature_inited = False

    def _init_feature(self):
        """
        load data
        :return:
        """
        self.text_feature = TextFeature(
            stopwords_path=self.stopwords_path,
            person_name_path=self.person_name_path,
            place_name_path=self.place_name_path,
            common_char_path=self.common_char_path,
            segment_sep=self.segment_sep
        )
        self.statistics_feature = StatisticsFeature(
            domain_sample_path=self.domain_sample_path,
            ngram=self.ngram,
            pmi_path=self.pmi_path,
            entropy_path=self.entropy_path)
        self.language_feature = LanguageFeature()
        self.feature_inited = True

    def check_feature_inited(self):
        """
        check if data loaded
        :return:
        """
        if not self.feature_inited:
            self._init_feature()

    def get_feature(self, query, is_word_segmented=False):
        """
        Get feature from query
        :param query: input query
        :param is_word_segmented: bool, is word segmented or not
        :return: features, terms
        """
        features = []
        terms = []

        self.check_feature_inited()
        text_terms, text_sent = self.text_feature.get_feature(query, is_word_segmented=is_word_segmented)
        stat_terms, stat_sent = self.statistics_feature.get_feature(query, is_word_segmented=is_word_segmented)
        lang_terms, lang_sent = self.language_feature.get_feature(query, is_word_segmented=is_word_segmented)
        # sentence feature
        text_sent.update(stat_sent)
        text_sent.update(lang_sent)
        logger.debug('sentence features: %s' % text_sent)
        sent_feature = [text_sent.query_length, text_sent.term_size, text_sent.ppl]
        # term feature
        for text, stat, lang in zip(text_terms, stat_terms, lang_terms):
            text.update(stat)
            text.update(lang)
            # logger.debug('term features: %s' % text)
            term_feature = [text.term_length, text.idx, text.offset, float(text.is_number),
                            float(text.is_chinese), float(text.is_alphabet), float(text.is_stopword),
                            float(text.is_name), float(text.is_common_char), text.embedding_sum, text.del_term_score,
                            text.idf, text.text_rank_score, text.tfidf_score, text.pmi_score, text.left_entropy_score,
                            text.right_entropy_score, text.del_term_ppl, text.term_ngram_score, text.left_term_score,
                            text.right_term_score]
            feature = sent_feature + term_feature
            features.append(feature)
            terms.append(text.term)
        logger.debug("[query]feature size: %s, term size: %s" % (len(features), len(terms)))

        return features, terms
