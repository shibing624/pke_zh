# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

KEA算法：
1. 规则过滤：1）候选短语限制最长三个词；2）候选短语不能是专名；3）候选短语开头和结尾不能是停用词
2. 提取关键词的tfidf特征
3. 用朴素贝叶斯计算候选关键词的得分排名，筛选top-k

"""

import os
import numpy as np
from jieba.analyse.tfidf import DEFAULT_IDF, IDFLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from pke_zh import USER_DATA_DIR
from pke_zh.utils.io_utils import load_pkl, save_pkl
from pke_zh.base import BaseKeywordExtractModel

default_model_path = os.path.join(USER_DATA_DIR, 'kea_model.pkl')


class Kea(BaseKeywordExtractModel):
    """Kea keyphrase extraction model."""

    def __init__(self, model_path=None, idf_path=None):
        """Redefining initializer for Kea."""
        super(Kea, self).__init__()
        self.instances = {}
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()
        # set the default model if none
        if model_path is None:
            self.model_path = default_model_path
        else:
            self.model_path = model_path

    def feature_scaling(self):
        """ Scale features to [0,1]. """
        candidates = self.instances.keys()
        X = [self.instances[u] for u in candidates]
        X = MinMaxScaler().fit_transform(X)
        for i, candidate in enumerate(candidates):
            self.instances[candidate] = X[i]

    def classify_candidates(self):
        """ Classify the candidates as keyphrase or not keyphrase.

            Args:
                model_path (str): the path to load the model in pickle format,
                    default to None.
        """
        # load the model
        clf = load_pkl(self.model_path)
        # get matrix of instances
        candidates = self.instances.keys()
        X = [self.instances[u] for u in candidates]
        # classify candidates
        y = clf.predict_proba(X)
        for i, candidate in enumerate(candidates):
            self.weights[candidate] = y[i][1]

    def candidate_selection(self):
        """Select 1-3 grams of `normalized` words as keyphrase candidates.
        Candidates that start or end with a stopword are discarded. Candidates
        that contain punctuation marks (from `string.punctuation`) as words are
        filtered out.
        """
        # select ngrams from 1 to 3 grams
        self.ngram_selection(n=3)
        # filter candidates
        self.candidate_filtering()
        # filter candidates that start or end with a stopword
        for k in list(self.candidates):
            # get the candidate
            v = self.candidates[k]
            # delete if candidate contains a stopword in first/last position
            words = [u.lower() for u in v.surface_forms[0]]
            if words[0] in self.stoplist or words[-1] in self.stoplist:
                del self.candidates[k]

    def feature_extraction(self, training=False):
        """Extract features for each keyphrase candidate. Features are the
        tf*idf of the candidate and its first occurrence relative to the
        document.

        Args:
            training (bool): indicates whether features are computed for the
                training set for computing IDF weights, defaults to false.
        """
        # initialize default document frequency counts if none provided

        # find the maximum offset
        maximum_offset = float(sum([s.length for s in self.sentences]))
        for k, v in self.candidates.items():
            # compute the tf*idf of the candidate
            # idf = math.log(N / candidate_df, 2)
            idf = self.idf_freq.get(k, self.median_idf)

            # add the features to the instance container
            self.instances[k] = np.array([len(v.surface_forms) * idf, v.offsets[0] / maximum_offset])
        # scale features
        self.feature_scaling()

    def candidate_weighting(self):
        """Extract features and classify candidates."""
        if not self.candidates:
            return

        self.feature_extraction()
        self.classify_candidates()

    def train(self, training_instances, training_classes):
        """ Train a Naive Bayes classifier and store the model in a file.

            Args:
                training_instances (list): list of features.
                training_classes (list): list of binary values.
        """
        clf = MultinomialNB()
        clf.fit(training_instances, training_classes)
        save_pkl(clf, self.model_path)

    def extract(self, input_file_or_string, n_best=10, pos=None):
        self.load_document(input=input_file_or_string, language='zh', normalization=None)
        self.candidate_selection()
        self.candidate_weighting()
        keyphrases = self.get_n_best(n=n_best)
        return keyphrases
