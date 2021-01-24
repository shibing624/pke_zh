# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 汉字处理的工具:判断unicode是否是汉字，数字，英文，或者其他字符。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


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
