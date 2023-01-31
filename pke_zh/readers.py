# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
modify from: https://github.com/boudinfl/pke/blob/master/pke/readers.py
"""
import re

import jieba
import jieba.posseg as pseg
import pandas as pd
from loguru import logger
from pke_zh.data_structures import Document

jieba.setLogLevel('ERROR')


def load_document_frequency_file(input_file, delimiter='\t'):
    """Load a tsv (tab-separated-values) file containing document frequencies.
    Automatically detects if input file is compressed (gzip) by looking at its
    extension (.gz).

    Args:
        input_file (str): the input file containing document frequencies in
            csv format.
        delimiter (str): the delimiter used for separating term-document
            frequencies tuples, defaults to '\t'.

    Returns:
        dict: a dictionary of the form {term_1: freq}, freq being an integer.
    """

    df_freq = pd.read_csv(input_file, sep=delimiter)
    df_freq['frequence'] = df_freq['frequence'].map(int)
    frequencies = df_freq.set_index('word').to_dict()['frequence']

    # return the populated dictionary
    return frequencies


def flatten_tuples(tuple_list):
    """
    :param tuple_list: list of list, each list contains tuple elements.
    :return: flatten tuple list
    """
    result = [t for tlist in tuple_list for t in tlist]
    return result


def fetch_nested_list_elements(nested_list, idx):
    """
    :param nested_list: list of list, each list is a tuple
    :param idx: target index in a tuple
    :return: list of list, each list contains target elements
    """
    result = [[t[idx] for t in inner_list] for inner_list in nested_list]
    return result


def reduce_charoffset(list_str):
    len_list = [len(x) for x in list_str]
    cur_sum = 0
    res = [0]
    for i in range(len(len_list) - 1):
        cur_sum += len_list[i]
        res.append(cur_sum)
    offset = [(x, x + len_list[i]) for i, x in enumerate(res)]
    return offset


def cut_sent(para):
    para = re.sub('([。！？\?;；])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？；;\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    paras = [p.strip() for p in para.split("\n") if p.strip()]
    return paras


class ChineseNLP(object):
    def __init__(self, max_length=1e6):
        self.max_length = max_length

    def __call__(self, text):
        """Split sentences
        Args:
            text(str): text for NLP Pipeline Processing.
        """
        clean_text = self.clean(text)
        sentences = cut_sent(clean_text)

        pseg_results, words, postags = self.sentence_pseg(sentences)

        char_offsets = list(map(lambda ws: reduce_charoffset(ws), words))

        # flatten char_offsets and make global index in the document.
        global_char_offsets = [char_offsets[0]]
        last_sent_offset = char_offsets[0][-1][1]
        for offsets in char_offsets[1:]:
            new_offset = [(x[0] + last_sent_offset, x[1] + last_sent_offset) for x in offsets]
            last_sent_offset = new_offset[-1][1]
            global_char_offsets.append(new_offset)
        global_char_offsets = flatten_tuples(global_char_offsets)

        result = [{'words': ws,
                   'POS': postags[i],
                   'char_offsets': char_offsets[i],
                   'global_char_offsets': global_char_offsets,
                   'word_POS': pseg_results}
                  for i, ws in enumerate(words)]
        return result

    def sentence_pseg(self, sentences):
        results = [[(s.word, s.flag) for s in pseg.cut(sent)] for sent in sentences]
        words = fetch_nested_list_elements(results, 0)
        postags = fetch_nested_list_elements(results, 1)
        return results, words, postags

    def clean(self, text):
        text = re.sub(r'img[_\d]+', '', text.strip())
        text = re.sub(r'_SEG_', '', text.strip())
        return text


class Reader(object):
    def read(self, path):
        raise NotImplementedError


class RawTextReader(Reader):
    """Reader for raw text."""

    def __init__(self, language='zh'):
        """Constructor for RawTextReader.

        Args:
            language (str): language of text to process.
        """

        self.language = language
        if self.language != 'zh':
            logger.warning('toolkit is created for zh language!')
        self.nlp = ChineseNLP()

    def read(self, text, **kwargs):
        """Read the input file and use spacy to pre-process.

        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy, default to 1,000,000 characters (1mb).
        """
        sentences = self.nlp(text)
        doc = Document.from_sentences(sentences,
                                      input_file=kwargs.get('input_file', None),
                                      **kwargs)

        return doc
