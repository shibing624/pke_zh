# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train model with xgboost
"""

import argparse

from wordrank import config
from wordrank.features.language_feature import LanguageFeature
from wordrank.features.statistics_feature import StatisticsFeature
from wordrank.features.text_feature import TextFeature


def data_reader(file_path, col_sep='\t'):
    contents = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split(col_sep)
            if len(parts) != 2:
                continue
            contents.append(parts[0])
            labels.append(parts[1])
    return contents, labels


def train(args):
    # 1.read train data
    contents, labels = data_reader(args.train_file, args.col_sep)
    print(contents)
    print(labels)
    # 2.get feature
    text_feature = TextFeature(
        stopwords_path=args.stopwords_path,
        person_name_path=args.person_name_path,
        place_name_path=args.place_name_path,
        common_char_path=args.common_char_path,
        segment_sep=args.segment_sep
    )
    statistics_feature = StatisticsFeature(
        domain_sample_path=args.domain_sample_path,
        ngram=args.ngram,
        pmi_path=args.pmi_path,
        entropy_path=args.entropy_path,
        segment_sep=args.segment_sep
    )
    language_feature = LanguageFeature(segment_sep=args.segment_sep)
    for content, label in zip(contents, labels):
        text_terms, text_sents = text_feature.get_feature(content, is_word_segmented=True)
        stat_terms, stat_sents = statistics_feature.get_feature(content, is_word_segmented=True)
        lang_terms, lang_sents = language_feature.get_feature(content, is_word_segmented=True)

        terms = []
        for text, stat, lang in zip(text_terms, stat_terms, lang_terms):
            text.update(stat)
            text.update(lang)
            terms.append(text)
        text_sents.update(stat_sents)
        text_sents.update(lang_sents)
        print('term features: %s' % terms)
        print('sentence features: %s' % text_sents)


    # 3.train classification model, save model file

    # 4.validation and evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_file', type=str, default=config.train_file, help='train file, file encode utf-8.')
    parser.add_argument('--test_file', type=str, default=config.test_file, help='the test file path.')
    parser.add_argument('--col_sep', type=str, default=config.col_sep, help='column sep')
    parser.add_argument('--segment_sep', type=str, default=config.segment_sep, help='segment_sep')
    parser.add_argument('--stopwords_path', type=str, default=config.stopwords_path, help='stop word file')
    parser.add_argument('--person_name_path', type=str, default=config.person_name_path, help='person name file')
    parser.add_argument('--place_name_path', type=str, default=config.place_name_path, help='place name file')
    parser.add_argument('--common_char_path', type=str, default=config.common_char_path, help='common_char_path')
    parser.add_argument('--domain_sample_path', type=str, default=config.domain_sample_path, help='domain_sample_path')
    parser.add_argument('--ngram', type=int, default=config.ngram, help='common_char_path')
    parser.add_argument('--pmi_path', type=str, default=config.pmi_path, help='pmi_path')
    parser.add_argument('--entropy_path', type=str, default=config.entropy_path, help='entropy_path')
    args = parser.parse_args()
    print(args)
    train(args)
