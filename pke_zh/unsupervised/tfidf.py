# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

TF-IDF keyphrase extraction model.

"""

from jieba.analyse.tfidf import DEFAULT_IDF
from pke_zh.base import BaseKeywordExtractModel


class TfIdf(BaseKeywordExtractModel):
    """TF*IDF keyphrase extraction model."""

    def __init__(self, idf_path=None, stopwords_path=None):
        """Redefining initializer for TopicRank."""
        super(TfIdf, self).__init__(stopwords_path)
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        idf_path = idf_path if idf_path is not None else DEFAULT_IDF
        self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'r', encoding='utf-8').read()
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf

    def candidate_selection(self, n=3, stoplist=None):
        """Select 1-3 grams as keyphrase candidates.

        :param n: int, the length of the n-grams, defaults to 3.
        :param stoplist: the stoplist for filtering candidates, defaults to
            `None`. Words that are punctuation marks from
            `string.punctuation` are not allowed.
        """
        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=n)

        # initialize empty list if stoplist is not provided
        if stoplist is not None:
            self.stoplist = stoplist

        # filter candidates containing punctuation marks
        self.candidate_filtering()

    def candidate_weighting(self):
        """Candidate weighting function using document frequencies."""
        total = sum([len(v.surface_forms) for v in self.candidates.values()])

        for k, v in self.candidates.items():
            # add the idf score to the weights container
            self.weights[k] = len(v.surface_forms) / total * self.idf_freq.get(k, self.median_idf)

    def extract(self, input_file_or_string, n_best=10, pos=None):
        self.load_document(input=input_file_or_string, language='zh')
        self.candidate_selection(n=3)
        self.candidate_weighting()
        keyphrases = self.get_n_best(n=n_best, redundancy_removal=True)
        return keyphrases
