# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 汉字处理的工具，判断unicode是否是汉字，数字，英文，或者其他字符
"""

import six
import numpy as np


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)


def is_number(uchar):
    """判断一个unicode是否是数字"""
    return u'u0030' <= uchar <= u'u0039'


def is_number_string(string):
    """判断是否全部是数字"""
    return all(is_number(c) for c in string)


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    return u'u0041' <= uchar <= u'u005a' or u'u0061' <= uchar <= u'u007a'


def is_alphabet_string(string):
    """判断是否全部为英文字母"""
    return all('a' <= c <= 'z' for c in string)


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    return not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar))


def char_similarity(str1, str2):
    if len(str1) <= 1 or len(str2) <= 1:
        return 0
    s1 = set(str1)
    s2 = set(str2)
    sim_score = len(s1 & s2) / max(len(s1), len(s2))
    return sim_score


def char_similarity2(str1, str2):
    """
    计算句子之间的相似度
    公式: similarity = |A∩B| / (log(|A|) + log(|B|))

    sim_score: float, 句子之间的相似度
    """
    if len(str1) <= 1 or len(str2) <= 1:
        return 0
    sim_score = len(set(str1) & set(str2)) / (np.log(len(str1)) + np.log(len(str2)))
    return sim_score


def edit_distance(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        import Levenshtein
        d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
    except:
        from difflib import SequenceMatcher
        # https://docs.python.org/2/library/difflib.html
        d = 1.0 - SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
    return d


if __name__ == '__main__':
    str1 = '好人一生啊'
    str2 = '好人一生平安'
    print(char_similarity(str1, str2))
    print(char_similarity2(str1, str2))
    print(edit_distance(str1, str2))
    print('edit sim:', 1 - edit_distance(str1, str2))

    str1 = '好人一生啊增加迭代轮次的同时增加迭代轮次前后的差值阈值'
    str2 = '好人一生平安是这个道理'
    print(char_similarity(str1, str2))
    print(char_similarity2(str1, str2))
    print(edit_distance(str1, str2))
    print('edit sim:', 1 - edit_distance(str1, str2))
