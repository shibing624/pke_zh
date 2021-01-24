# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

* 文本特征：包括Query长度、Term长度，Term在Query中的偏移量，term词性、长度信息、
term数目、位置信息、句法依存tag、是否数字、是否英文、是否停用词、是否专名实体、
是否重要行业词、embedding模长、删词差异度、以及短语生成树得到term权重等。

"""

import codecs
from copy import deepcopy

import numpy as np
from text2vec import Vector, Similarity
from text2vec.similarity import SimType

from wordrank import config
from wordrank.utils.logger import logger
from wordrank.utils.text_utils import is_number_string, is_alphabet_string, is_chinese_string
from wordrank.utils.tokenizer import word_segment


class AttrDict(dict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TextFeature(object):
    def __init__(self,
                 stopwords_path=config.stopwords_path,
                 person_name_path=config.person_name_path,
                 place_name_path=config.place_name_path,
                 common_char_path=config.common_char_path,
                 segment_sep=config.segment_sep
                 ):
        self.stopwords = self.load_set_file(stopwords_path)
        self.person_names = self.load_set_file(person_name_path)
        self.place_names = self.load_set_file(place_name_path)
        self.common_chars = self.load_set_file(common_char_path)
        self.segment_sep = segment_sep
        self.vec = Vector()
        self.sim = Similarity(similarity_type=SimType.WMD)

    @staticmethod
    def load_set_file(path):
        words = set()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w.split()[0])
        return words

    def is_stopword(self, word):
        return word in self.stopwords

    def is_name(self, word):
        names = self.person_names | self.place_names
        return word in names

    def is_entity(self, pos, entity_pos=('ns', 'n', 'vn', 'v')):
        return pos in entity_pos

    def is_common_char(self, c):
        return c in self.common_chars

    def is_common_char_string(self, word):
        return all(self.is_common_char(c) for c in word)

    def get_feature(self, query, is_word_segmented=False):
        """
        Get text feature
        :param query:
        :param is_word_segmented:
        :return: list, list: term features, sentence features
        """
        term_features = []
        if is_word_segmented:
            word_seq = query.split(self.segment_sep)
        else:
            word_seq = word_segment(query, cut_type='word', pos=False)
        logger.debug('%s' % word_seq)

        # sentence
        sentence_features = AttrDict(
            query_length=len(query),
            term_size=len(word_seq),
        )

        # term
        idx = 0
        offset = 0
        for word in word_seq:
            emb = self.vec.encode(word)
            word_list = deepcopy(word_seq)
            if word in word_list:
                word_list.remove(word)
            del_word_query = ''.join(word_list)
            del_term_sim_score = self.sim.get_score(query, del_word_query)
            term_features.append(AttrDict(
                term=word,
                term_length=len(word),
                idx=idx,
                offset=offset,
                is_number=is_number_string(word),
                is_chinese=is_chinese_string(word),
                is_alphabet=is_alphabet_string(word),
                is_stopword=self.is_stopword(word),
                is_name=self.is_name(word),
                # is_entity=self.is_entity(pos),
                is_common_char=self.is_common_char_string(word),
                embedding_sum=np.sum(emb),
                del_term_score=del_term_sim_score,
            ))
            idx += len(word)
            offset += 1

        return term_features, sentence_features
