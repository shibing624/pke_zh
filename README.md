# pke_zh
[![PyPI version](https://badge.fury.io/py/pke_zh.svg)](https://badge.fury.io/py/pke_zh)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/pke_zh.svg)](https://github.com/shibing624/pke_zh/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/pke_zh.svg)](https://github.com/shibing624/pke_zh/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

PKE_zh, Python Keyphrase Extraction for ZH(chinese).

**pke_zh**实现了多种中文关键词提取算法，包括有监督的WordRank、无监督的TextRank, TfIdf, KeyBert, PositionRank, TopicRank等，扩展性强，开箱即用。


**Guide**

- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Reference](#reference)

# Feature

如何提取query或者文档的关键词？


### 有监督方法
#### 特征工程的解决思路
1. 实现时采用模型打分方法，以搜索query为原始语料，人工标注句子中各词重要度

> 重要度共分4级：
> * Super important：主要包括POI核心词，比如“方特、欢乐谷”
> * Required：包括行政区词、品类词等，比如“北京 温泉”中“北京”和“温泉”都很重要
> * Important：包括品类词、门票等，比如“顺景 温泉”中“温泉”相对没有那么重要，用户搜“顺景”大部分都是温泉的需求
> * Unimportant：包括语气词、代词、泛需求词、停用词等

上例中可见“温泉”在不同的query中重要度是不同的。

**特征方法**

* 文本特征：包括Query长度、Term长度，Term在Query中的偏移量，term词性、长度信息、term数目、位置信息、句法依存tag、是否数字、是否英文、是否停用词、是否专名实体、是否重要行业词、embedding模长、删词差异度、以及短语生成树得到term权重等
* 统计特征：包括PMI、IDF、textrank值、前后词互信息、左右邻熵、独立检索占比（term单独作为query的qv/所有包含term的query的qv和）、统计概率、idf变种iqf
* 语言模型特征：整个query的语言模型概率 / 去掉该Term后的Query的语言模型概率

2. 模型方面采用树模型（XGBoost等）进行训练，得到权重分类模型后在线上预测
![term-weighting](./docs/gbdt.png)

#### 深度模型的解决思路
* 利用深度学习模型来学习term重要性，比如通过训练基于BiLSTM+Attention的query意图分类模型
* 基于Seq2Seq/Transformer训练的query翻译改写模型得到的attention权重副产物再结合其他策略或作为上述分类回归模型的特征也可以用于衡量term的重要性
* 利用BERT模型训练端到端的词分级模型，类似序列标注模型，后接CRF判定词重要性权重输出


**深度模型**

* BERT CLS + softmax 
* Seq2Seq 文本摘要模型

### 无监督方法
- [x] TextRank
- [x] TfIdf
- [x] SingleRank
- [x] PositionRank
- [x] TopicRank
- [x] MultipartiteRank
- [x] Yake
- [x] KeyBert



# Install
* From pip:
```shell
pip3 install -U pke_zh
```
* From source：
```shell
git clone https://github.com/shibing624/pke_zh.git
cd pke_zh
python3 setup.py install
```

### 依赖数据
* 千兆中文文本训练的语言模型[zh_giga.no_cna_cmn.prune01244.klm(2.8G)](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm)，模型由pycorrector库自动下载于：~/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm 。
* 中文文本匹配模型[shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese) ，模型由transformers库自动下载于：~/.cache/huggingface/transformers/ 下。

# Usage

## 有监督关键词提取

直接调用训练好的WordRank模型，模型自动下载本地`~/.cache/pke_zh/wordrank_model.pkl`

example: [examples/keyphrase_extraction_demo.py](examples/keyphrase_extraction_demo.py)

```python
from pke_zh.supervised.wordrank import WordRank
m = WordRank()
print(m.extract("哪里下载电视剧周恩来？"))
```

output:
```shell
[('电视剧', 3), ('周恩来', 3), ('下载', 2), ('哪里', 1), ('？', 0)]
```
> 3：核心词；2：限定词；1：可省略词；0：干扰词。

### 基于自有数据训练模型

训练example: [examples/train_supervised_wordrank_demo.py](examples/train_supervised_wordrank_demo.py)


## 无监督关键词提取
支持TextRank、TfIdf、KeyBert等关键词提取算法。

example: [examples/unsupervised_demo.py](examples/unsupervised_demo.py)


```python
from pke_zh.unsupervised.textrank import TextRank
from pke_zh.unsupervised.tfidf import TfIdf
from pke_zh.unsupervised.singlerank import SingleRank
from pke_zh.unsupervised.positionrank import PositionRank
from pke_zh.unsupervised.topicrank import TopicRank
from pke_zh.unsupervised.multipartiterank import MultipartiteRank
from pke_zh.unsupervised.yake import Yake
from pke_zh.unsupervised.keybert import KeyBert

q = '哪里下载电视剧周恩来？'
TextRank_m = TextRank()
TfIdf_m = TfIdf()
PositionRank_m = PositionRank()
KeyBert_m = KeyBert()

r = TextRank_m.extract(q)
print('TextRank:', r)

r = TfIdf_m.extract(q)
print('TfIdf:', r)

r = PositionRank_m.extract(q)
print('PositionRank_m:', r)

r = KeyBert_m.extract(q)
print('KeyBert_m:', r)
```

output:
```shell
TextRank: [('电视剧', 1.00000002)]
TfIdf: [('哪里下载', 1.328307500322222), ('下载电视剧', 1.328307500322222), ('电视剧周恩来', 1.328307500322222)]
PositionRank_m: [('电视剧', 1.0)]
KeyBert_m: [('电视剧', 0.47165293)]
```


# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/pke_zh.svg)](https://github.com/shibing624/pke_zh/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624*, 备注：*姓名-公司名-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了pke_zh，请按如下格式引用：
APA:
```latex
Xu, M. pke_zh: Python keyphrase extraction toolkit for chinese (Version 0.2.2) [Computer software]. https://github.com/shibing624/pke_zh
```

BibTeX:
```latex
@misc{pke_zh,
  author = {Xu, Ming},
  title = {pke_zh: Python keyphrase extraction toolkit for chinese},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shibing624/pke_zh}},
}
```

# License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加pke_zh的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


# Reference

- [boudinfl/pke](https://github.com/boudinfl/pke)
- [Context-Aware Document Term Weighting for Ad-Hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/TheWebConf_2020_Dai.pdf)
- [term weighting](https://zhuanlan.zhihu.com/p/90957854)
- [DeepCT](https://github.com/AdeDZY/DeepCT)
