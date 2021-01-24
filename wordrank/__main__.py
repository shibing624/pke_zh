# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: cmd line
"""

import argparse

from wordrank import config
from wordrank.model import predict
from wordrank.model import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action="store_true", default=False, help='train or predict')
    parser.add_argument('--train_file', type=str, default=config.train_file, help='train file, file encode utf-8.')
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
    parser.add_argument('--model_path', type=str, default=config.model_path, help='model file path to save')
    parser.add_argument('--query', type=str, default='井冈山景点介绍', help='input query')
    args = parser.parse_args()
    print(args)
    if args.train:
        train(args.train_file,
              args.col_sep,
              args.stopwords_path,
              args.person_name_path,
              args.place_name_path,
              args.common_char_path,
              args.segment_sep,
              args.domain_sample_path,
              args.ngram,
              args.pmi_path,
              args.entropy_path,
              args.model_path)
    else:
        predict(args.query,
                args.model_path,
                args.stopwords_path,
                args.person_name_path,
                args.place_name_path,
                args.common_char_path,
                args.segment_sep,
                args.domain_sample_path,
                args.ngram,
                args.pmi_path,
                args.entropy_path)
