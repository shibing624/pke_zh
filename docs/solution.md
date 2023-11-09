## Features

如何提取query或者文档的关键词？


### 有监督方法
#### 特征工程的解决思路
把关键词提取任务转化为分类任务，对输入query句子分词并提取多种特征，再把特征喂给机器学习模型，模型区分出各词的重要性得分，这样挑出topK个词作为关键词。

* 文本特征：包括Query长度、Term长度，Term在Query中的偏移量，term词性、长度信息、term数目、位置信息、句法依存tag、是否数字、是否英文、是否停用词、是否专名实体、是否重要行业词、embedding模长、删词差异度、以及短语生成树得到term权重等
* 统计特征：包括PMI、IDF、TextRank值、前后词互信息、左右邻熵、独立检索占比（term单独作为query的qv/所有包含term的query的qv和）、统计概率、idf变种iqf
* 语言模型特征：整个query的语言模型概率 / 去掉该Term后的Query的语言模型概率


训练样本形如：
```shell
邪御天娇 免费 阅读,3 1 1
```

**重要度label**共分4级：
- Super important：3级，主要包括POI核心词，比如“方特、欢乐谷”
- Required：2级，包括行政区词、品类词等，比如“北京 温泉”中“北京”和“温泉”都很重要
- Important：1级，包括品类词、门票等，比如“顺景 温泉”中“温泉”相对没有那么重要，用户搜“顺景”大部分都是温泉的需求
- Unimportant：0级，包括语气词、代词、泛需求词、停用词等

分类模型可以是GBDT、LR、SVM、Xgboost等，这里以GBDT为例，GBDT模型（WordRank）的输入是特征向量，输出是重要度label。

![term-weighting](https://github.com/shibing624/pke_zh/blob/main/docs/gbdt.png)

### 深度模型的解决思路
- 思路一：本质依然是把关键词提取任务转化为词重要度分类任务，利用深度模型学习term重要度，取代人工提取特征，模型端到端预测词重要度label，按重要度排序后挑出topK个词作为关键词。深度模型有TextCNN、Fasttext、Transformer、BERT等，适用于分类任务的模型都行。分类任务实现参考：https://github.com/shibing624/pytextclassifier
- 思路二：用Seq2Seq生成模型，输入query，输出关键词或者摘要，生成模型可以是T5、Bart、Seq2Seq等，生成任务实现参考：https://github.com/shibing624/textgen

## 无监督方法
- 统计算法
- [x] TFIDF，是很强的baseline，有较强普适性，基本能应付大部分关键词抽取场景，简单有效，速度很快，效果一般
- [x] YAKE，人工总结规则的方法，不依赖外部语料，从单文档提取关键词，速度很快，效果差
- 图算法
- [x] TextRank，简单套用PageRank思想到关键词提取的方法，效果不比TFIDF强，而且涉及网络构建和随机游走迭代，速度慢，效果一般
- [x] SingleRank，类似TextRank，是PageRank的变体，可以提取出关键短语，速度快，效果一般
- [x] TopicRank，基于主题模型的关键词提取算法，考虑了文档中词语的语义关系，可以提取出与文档主题相关的关键词，速度慢，效果一般
- [x] MultipartiteRank，一种基于多元关系的关键词提取算法，在TopicRank的基础上，考虑了词语的语义关系和词语位置，速度慢，效果一般
- [x] PositionRank，是基于PageRank的图关系计算词权重，考虑了词位置和词频，速度一般，效果好
- 语义模型
- [x] KeyBERT，利用了预训练语言模型的能力来提取关键词，速度很慢，效果最好

**模型选型**
- 要求速度快，选择TFIDF、YAKE、PositionRank
- 要求效果好，选择KeyBERT

无监督算法介绍见文章[中文关键词提取算法](https://blog.csdn.net/mingzai624/article/details/129012015)

### 依赖数据
* 千兆中文文本训练的语言模型[zh_giga.no_cna_cmn.prune01244.klm(2.8G)](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm)，模型由pycorrector库自动下载于 `~/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm` 。
* 中文文本匹配模型[shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese) ，模型由transformers库自动下载于 `~/.cache/huggingface/transformers/` 下。


