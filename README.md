[**🇨🇳中文**](https://github.com/shibing624/pke_zh/blob/main/README.md) |  [**📖文档/Docs**](https://github.com/shibing624/pke_zh/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/pke_zh">
    <img src="https://github.com/shibing624/pke_zh/blob/main/docs/pke_zh.png" alt="Logo" height="156">
  </a>
</div>

-----------------

# pke_zh: Python Keyphrase Extraction for zh(chinese)
[![PyPI version](https://badge.fury.io/py/pke_zh.svg)](https://badge.fury.io/py/pke_zh)
[![Downloads](https://static.pepy.tech/badge/pke_zh)](https://pepy.tech/project/pke_zh)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/pke_zh.svg)](https://github.com/shibing624/pke_zh/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/pke_zh.svg)](https://github.com/shibing624/pke_zh/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)


PKE_zh, Python Keyphrase Extraction for zh(chinese).

**pke_zh**实现了多种中文关键词提取算法，包括有监督的WordRank，无监督的TextRank、TfIdf、KeyBert、PositionRank、TopicRank等，扩展性强，开箱即用。


**Guide**

- [Features](#Features)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [References](#references)

## Features
#### 有监督方法
- [x] WordRank：本项目基于Python实现了句子的文本特征、统计特征、Tag特征、语言模型特征提取，结合GBDT模型区分出句子中各词的重要性得分，进而提取关键词，速度快，效果好，泛化性一般，依赖有监督数据。
#### 无监督方法
- 统计算法
- [x] TFIDF：本项目基于jieba的IDF词表实现了TFIDF的关键词抽取，该方法是很强的baseline，有较强普适性，基本能应付大部分关键词抽取场景，简单有效，速度很快，效果一般
- [x] YAKE：本项目实现了YAKE，该算法基于人工总结的规则（词的位置，词频，上下文关系，词在句中频率），不依赖外部语料，从单文档提取关键词，速度很快，效果差
- 图算法
- [x] TextRank：本项目基于networkx实现了TextRank，该算法简单套用PageRank思想到关键词提取，效果不比TFIDF强，而且涉及网络构建和随机游走迭代，速度慢，效果一般
- [x] SingleRank：本项目基于networkx实现了SingleRank，该算法类似TextRank，是PageRank的变体，可以提取出关键短语，速度快，效果一般
- [x] TopicRank：本项目基于networkx实现了TopicRank，该算法基于主题模型的关键词提取，考虑了文档中词语的语义关系，可以提取出与文档主题相关的关键词，速度慢，效果一般
- [x] MultipartiteRank：本项目基于networkx实现了MultipartiteRank，该算法基于多元关系提取关键词，在TopicRank的基础上，考虑了词语的语义关系和词语位置，速度慢，效果一般
- [x] PositionRank：本项目基于networkx实现了PositionRank，该算法基于PageRank的图关系计算词权重，考虑了词位置和词频，速度一般，效果好
- 语义模型
- [x] KeyBERT：本项目基于text2vec实现了KeyBert，利用了预训练句子表征模型计算句子embedding和各词embedding相似度来提取关键词，速度很慢，效果最好

- 延展阅读：[中文关键词提取解决思路](https://github.com/shibing624/pke_zh/blob/main/docs/solution.md)

**模型选型**
- 要求速度快，选择TFIDF、PositionRank、WordRank
- 要求效果好，选择KeyBERT
- 有监督数据，选择WordRank



## Install
* From pip:
```zsh
pip install -U pke_zh
```

* From source：
```zsh
git clone https://github.com/shibing624/pke_zh.git
cd pke_zh
python setup.py install
```

## Usage

### 有监督关键词提取

#### pke_zh快速预测
example: [examples/keyphrase_extraction_demo.py](examples/keyphrase_extraction_demo.py)

```python
from pke_zh import WordRank
m = WordRank()
print(m.extract("哪里下载电视剧周恩来？"))
```

output:
```shell
[('电视剧', 3), ('周恩来', 3), ('下载', 2), ('哪里', 1), ('？', 0)]
```
- 返回值：核心短语列表，(keyphrase, score)，其中score： 3：核心词；2：限定词；1：可省略词；0：干扰词 
- **score**共分4级：
  - Super important：3级，主要包括POI核心词，比如“方特、欢乐谷”
  - Required：2级，包括行政区词、品类词等，比如“北京 温泉”中“北京”和“温泉”都很重要
  - Important：1级，包括品类词、门票等，比如“顺景 温泉”中“温泉”相对没有那么重要，用户搜“顺景”大部分都是温泉的需求
  - Unimportant：0级，包括语气词、代词、泛需求词、停用词等
- 模型：默认调用训练好的WordRank模型[wordrank_model.pkl](https://github.com/shibing624/pke_zh/releases/tag/0.2.2)，模型自动下载于 `~/.cache/pke_zh/wordrank_model.pkl`
#### 训练模型
**WordRank模型**：对输入query分词并提取多类特征，再把特征喂给GBDT等分类模型，模型区分出各词的重要性得分，挑出topK个词作为关键词

* 文本特征：包括Query长度、Term长度，Term在Query中的偏移量，term词性、长度信息、term数目、位置信息、句法依存tag、是否数字、是否英文、是否停用词、是否专名实体、是否重要行业词、embedding模长、删词差异度、以及短语生成树得到term权重等
* 统计特征：包括PMI、IDF、TextRank值、前后词互信息、左右邻熵、独立检索占比（term单独作为query的qv/所有包含term的query的qv和）、统计概率、idf变种iqf
* 语言模型特征：整个query的语言模型概率 / 去掉该Term后的Query的语言模型概率


训练样本格式：
```shell
邪御天娇 免费 阅读,3 1 1
```
模型结构：

![term-weighting](https://github.com/shibing624/pke_zh/blob/main/docs/gbdt.png)

training example: [examples/train_supervised_wordrank_demo.py](examples/train_supervised_wordrank_demo.py)

### 无监督关键词提取
支持TextRank、TfIdf、PositionRank、KeyBert等关键词提取算法。

example: [examples/unsupervised_demo.py](examples/unsupervised_demo.py)


```python
from pke_zh import TextRank, TfIdf, SingleRank, PositionRank, TopicRank, MultipartiteRank, Yake, KeyBert
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

### 无监督关键句提取（自动摘要）
支持TextRank摘要提取算法。

example: [examples/keysentences_extraction_demo.py](examples/keysentences_extraction_demo.py)


```python
from pke_zh import TextRank
m = TextRank()
r = m.extract_sentences("较早进入中国市场的星巴克，是不少小资钟情的品牌。相比 在美国的平民形象，星巴克在中国就显得“高端”得多。用料并无差别的一杯中杯美式咖啡，在美国仅约合人民币12元，国内要卖21元，相当于贵了75%。  第一财经日报")
print(r)
```

output:
```shell
[('相比在美国的平民形象', 0.13208935993025409), ('在美国仅约合人民币12元', 0.1320761453200497), ('星巴克在中国就显得“高端”得多', 0.12497451534612379), ('国内要卖21元', 0.11929080110899569) ...]
```

## Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/pke_zh.svg)](https://github.com/shibing624/pke_zh/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624*, 备注：*姓名-公司名-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


## Citation

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

## License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加pke_zh的链接和授权协议。


## Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


## References

- [boudinfl/pke](https://github.com/boudinfl/pke)
- [Context-Aware Document Term Weighting for Ad-Hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/TheWebConf_2020_Dai.pdf)
- [term weighting](https://zhuanlan.zhihu.com/p/90957854)
- [DeepCT](https://github.com/AdeDZY/DeepCT)
