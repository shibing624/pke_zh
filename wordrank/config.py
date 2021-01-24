# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))
# where to save custom files
extra_data_dir = os.path.join(pwd_path, "../extra_data")

col_sep = ','  # separate label and content of train data
segment_sep = ' '  # separate cut word
train_file = os.path.join(extra_data_dir, "train.csv")
test_file = os.path.join(extra_data_dir, "test.csv")

sentence_delimiters = ['？', '！', '。', '；', '……', '…']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

# inner data file
stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')
person_name_path = os.path.join(pwd_path, 'data/person_name.txt')
place_name_path = os.path.join(pwd_path, 'data/place_name.txt')
common_char_path = os.path.join(pwd_path, 'data/common_char_set.txt')

# custom domain file for statistics
domain_sample_path = os.path.join(extra_data_dir, 'train.csv')
pmi_path = os.path.join(extra_data_dir, 'pmi_word_score.json')
entropy_path = os.path.join(extra_data_dir, 'entropy_word_score.json')
# word_vocab_path =os.path.join(extra_data_dir, 'word_vocab.txt')
# feature_vec_path = os.path.join(extra_data_dir, 'feature_vec.pkl')

ngram = 4

model_path = os.path.join(extra_data_dir, 'classify_model.pkl')
if not os.path.exists(extra_data_dir):
    os.makedirs(extra_data_dir)
