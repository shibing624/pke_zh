# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train model with xgboost
"""

import argparse
import sys

sys.path.append('..')
from pke_zh.wordrank import WordRank

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Whether not to train")
    parser.add_argument("--do_predict", action="store_true", help="Whether not to predict")
    parser.add_argument('--train_file', type=str, default='data/train.csv', help='train file, file encode utf-8.')
    parser.add_argument('--test_file', type=str, default='data/test.csv', help='the test file path.')
    parser.add_argument('--col_sep', type=str, default=',', help='column sep')
    parser.add_argument('--segment_sep', type=str, default=' ', help='segment_sep')
    parser.add_argument('--ngram', type=int, default=4, help='common_char_path')
    parser.add_argument('--model_path', type=str, default='wordrank_test.pkl', help='model file path to save')
    args = parser.parse_args()
    print(args)
    m = WordRank(model_path=args.model_path)
    if args.do_train:
        m.train(args.train_file, args.col_sep, is_word_segmented=True)
    if args.do_predict:
        pred_result = m.predict('井冈山景点介绍', is_word_segmented=False)
        print("predict label: %s" % pred_result)
