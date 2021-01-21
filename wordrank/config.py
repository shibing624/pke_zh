# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

input_file_path = os.path.join(pwd_path, "../extra_data/samples.txt")

sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；',';', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

# stopwords
stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')

col_sep = '\t'  # separate label and content of train data

# active learning params
output_dir = os.path.join(pwd_path, "../extra_data")  # where to save outputs

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
