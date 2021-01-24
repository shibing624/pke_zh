# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Word Rank module, main
"""

from wordrank import config
from wordrank.feature import Feature
from wordrank.utils import tokenizer
from wordrank.utils.io_utils import load_pkl
from wordrank.utils.logger import logger
from wordrank.utils.text_utils import convert_to_unicode


class WordRank(Feature):
    def __init__(self,
                 model_path=config.model_path,
                 stopwords_path=config.stopwords_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 common_char_path=config.common_char_path,
                 segment_sep = config.segment_sep,
                 domain_sample_path=config.domain_sample_path,
                 ngram=config.ngram,
                 pmi_path=config.pmi_path,
                 entropy_path=config.entropy_path,
                 sentence_delimiters=config.sentence_delimiters,
                 ):
        super(WordRank, self).__init__(
            stopwords_path=stopwords_path,
            person_name_path=person_name_path,
            place_name_path=place_name_path,
            common_char_path=common_char_path,
            segment_sep = segment_sep,
            domain_sample_path=domain_sample_path,
            ngram=ngram,
            pmi_path=pmi_path,
            entropy_path=entropy_path,
            sentence_delimiters=sentence_delimiters
        )
        self.model_path = model_path
        self.inited = False

    def _init(self):
        self.model = load_pkl(self.model_path)
        self.inited = True

    def check_inited(self):
        if not self.inited:
            self._init()

    def rank_query(self, query):
        self.check_inited()
        # get feature
        data_feature, terms = self.get_feature(query, is_word_segmented=False)
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
