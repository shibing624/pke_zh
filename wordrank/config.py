# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

input_file_path = os.path.join(pwd_path, "../extra_data/samples.txt")

# segment_type optionals: "word, char"
segment_type = 'word'
# feature_type optionals: "tfidf, tf"
feature_type = 'tfidf'

seg_input_file_path = os.path.join(pwd_path, "../extra_data/samples_seg_{}.txt".format(segment_type))
col_sep = '\t'  # separate label and content of train data
num_classes = 6

# active learning params
output_dir = os.path.join(pwd_path, "../extra_data")  # where to save outputs

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
