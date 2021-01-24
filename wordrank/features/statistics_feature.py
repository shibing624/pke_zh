# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

* 统计特征：包括PMI、IDF、textrank值、前后词互信息、左右邻熵、
独立检索占比（term单独作为query的qv/所有包含term的query的qv和）、统计概率TF、idf变种iqf
"""

from wordrank import config
from wordrank.features.pmi import PMI
from wordrank.features.text_feature import AttrDict
from wordrank.features.textrank import TextRank
from wordrank.features.tfidf import TFIDF
from wordrank.utils.tokenizer import word_segment


class StatisticsFeature(object):
    def __init__(self,
                 domain_sample_path=config.domain_sample_path,
                 ngram=4,
                 pmi_path=config.pmi_path,
                 entropy_path=config.entropy_path,
                 segment_sep=config.segment_sep
                 ):
        self.tfidf = TFIDF()
        self.text_rank = TextRank()
        self.segment_sep = segment_sep
        self.pmi_model = PMI(text=self.read_text(domain_sample_path),
                             ngram=ngram,
                             pmi_path=pmi_path,
                             entropy_path=entropy_path)

    def _get_tags_score(self, word, tags):
        score = 0.0
        for i, s in tags:
            if word == i:
                score = s
        return score

    @staticmethod
    def read_text(file_path, col_sep=config.col_sep, limit_len=100000):
        text = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split(col_sep)
                text += parts[0]
        if limit_len > 0:
            result = text[:limit_len]
        else:
            result = text
        return result

    def get_feature(self, query, is_word_segmented=False):
        """
        Get statistics feature
        :param query:
        :param is_word_segmented:
        :return: list, list: term features, sentence features
        """
        term_features = []
        sentence_features = {}
        rank_tags = self.text_rank.textrank(query, withWeight=True)
        tfidf_tags = self.tfidf.extract_tags(query, withWeight=True)

        if is_word_segmented:
            word_seq = query.split(self.segment_sep)
        else:
            word_seq = word_segment(query, cut_type='word', pos=False)

        count = 0
        for word in word_seq:
            idf = self.tfidf.idf_freq.get(word, self.tfidf.median_idf)
            # PMI & Entropy
            left_word = word_seq[count - 1] if count > 0 else ''
            right_word = word_seq[count + 1] if count < len(word_seq) - 1 else ''
            left_right_word = ''.join([left_word, right_word])
            entropy_score = self.pmi_model.entropy_score(left_right_word)
            term_features.append(AttrDict(
                idf=idf,
                text_rank_score=self._get_tags_score(word, rank_tags),
                tfidf_score=self._get_tags_score(word, tfidf_tags),
                pmi_score=self.pmi_model.pmi_score(left_right_word),
                left_entropy_score=entropy_score[0],
                right_entropy_score=entropy_score[1],
            ))
            count += 1

        return term_features, sentence_features
