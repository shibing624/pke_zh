# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

from setuptools import setup, find_packages

__version__ = None
exec(open('pke_zh/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for pke_zh.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='pke_zh',
    version=__version__,
    description='pke_zh, context-aware bag-of-words term weights for query and document.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/pke_zh',
    license="Apache 2.0",
    zip_safe=False,
    python_requires='>=3.5',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='pke_zh,term weighting,textrank,word rank,wordweight',
    install_requires=[
        "loguru",
        "jieba",
        "pandas",
        "numpy",
        "six",
        "tqdm",
        "requests",
        "scikit-learn",
        "scipy",
        "networkx",
        "text2vec",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'pke_zh': 'pke_zh'},
    package_data={'pke_zh': ['*.*', 'data/*', ]}
)
