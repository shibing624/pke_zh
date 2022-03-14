# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from wordrank.version import __version__
from wordrank import config
from wordrank.wordrank import WordRank
from wordrank.feature import Feature
from wordrank.features.text_feature import TextFeature
from wordrank.features.textrank import TextRank4Keyword
from wordrank.features.tfidf import TFIDF4Keyword
from wordrank.features.language_feature import LanguageFeature
from wordrank.features.pmi import PMI
from wordrank.features.statistics_feature import StatisticsFeature
from wordrank.utils.logger import set_log_level

WR = WordRank()
rank = WR.rank
