# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from pke_zh.version import __version__

USER_DATA_DIR = os.path.expanduser('~/.cache/pke_zh/')
os.makedirs(USER_DATA_DIR, exist_ok=True)

from pke_zh import unsupervised
from pke_zh import supervised
