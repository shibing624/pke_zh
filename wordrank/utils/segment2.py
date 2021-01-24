# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import codecs

import jieba.posseg

from wordrank import config

class AttrDict(dict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



class WordSegmentation(object):
    """ 分词 """

    def __init__(self, stop_words_file=config.stopwords_path, allow_speech_tags=config.allow_speech_tags):
        """
        Keyword arguments:
        stop_words_file    -- 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
        allow_speech_tags  -- 词性列表，用于过滤
        """

        allow_speech_tags = [item for item in allow_speech_tags]

        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = set()
        self.stop_words_file = stop_words_file
        if type(stop_words_file) is str:
            self.stop_words_file = stop_words_file
        for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
            self.stop_words.add(word.strip())

    def segment(self, text, lower=True, use_stop_words_filter=True, use_speech_tags_filter=False):
        """对一段文本进行分词，返回list类型的分词结果
        Keyword arguments:
        lower                  -- 是否将单词小写（针对英文）
        use_stop_words_filter         -- 若为True，则利用停止词集合来过滤（去掉停止词）
        use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。
        """
        jieba_result = jieba.posseg.cut(text)

        if use_speech_tags_filter:
            jieba_result = [w for w in jieba_result if w.flag in self.default_speech_tag_filter]
        else:
            jieba_result = [w for w in jieba_result]

        # 去除特殊符号
        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]

        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words_filter:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list

    def segment_sentences(self, sentences, lower=True, use_stop_words_filter=True, use_speech_tags_filter=False):
        """将列表sequences中的每个元素/句子转换为由单词构成的列表。

        sequences -- 列表，每个元素是一个句子（字符串类型）
        """
        res = []
        for sentence in sentences:
            res.append(self.segment(text=sentence,
                                    lower=lower,
                                    use_stop_words_filter=use_stop_words_filter,
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res


class SentenceSegmentation(object):
    """ 分句 """

    def __init__(self, delimiters=config.sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        self.delimiters = set([item for item in delimiters])

    def segment(self, text):
        res = [text]

        for sep in self.delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res


class Segmentation(object):

    def __init__(self,
                 stopwords_file=config.stopwords_path,
                 allow_speech_tags=config.allow_speech_tags,
                 delimiters=config.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ws = WordSegmentation(stop_words_file=stopwords_file, allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimiters)

    def segment(self, text, lower=False):
        sentences = self.ss.segment(text)
        words = self.ws.segment_sentences(sentences=sentences,
                                          lower=lower,
                                          use_stop_words_filter=False,
                                          use_speech_tags_filter=False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences,
                                                        lower=lower,
                                                        use_stop_words_filter=True,
                                                        use_speech_tags_filter=False)

        words_all_filters = self.ws.segment_sentences(sentences=sentences,
                                                      lower=lower,
                                                      use_stop_words_filter=True,
                                                      use_speech_tags_filter=True)

        return AttrDict(
            sentences=sentences,
            words=words,
            words_no_stop_words=words_no_stop_words,
            words_all_filters=words_all_filters
        )
