# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Word Rank module, main
"""
import codecs
import os
from loguru import logger
import re
from collections import Counter
from typing import Optional
from copy import deepcopy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pke_zh import USER_DATA_DIR
from pke_zh.utils.io_utils import load_pkl, save_pkl, save_json, load_json
from pke_zh.utils.text_utils import convert_to_unicode, is_number_string, is_alphabet_string, is_chinese_string
from pke_zh.utils.tokenizer import word_segment
from pke_zh.utils.file_utils import get_file
from pke_zh.unsupervised.tfidf import TfIdf
from pke_zh.unsupervised.textrank import TextRank

pwd_path = os.path.abspath(os.path.dirname(__file__))

# inner data file
default_stopwords_path = os.path.join(pwd_path, '../data/stopwords.txt')
person_name_path = os.path.join(pwd_path, '../data/person_name.txt')
place_name_path = os.path.join(pwd_path, '../data/place_name.txt')
common_char_path = os.path.join(pwd_path, '../data/common_char_set.txt')
pmi_path = os.path.join(pwd_path, '../data/pmi_word_score.json')
entropy_path = os.path.join(pwd_path, '../data/entropy_word_score.json')

# word rank model path
default_model_path = os.path.join(USER_DATA_DIR, 'wordrank_model.pkl')


class AttrDict(dict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class StatisticsFeature:
    """
    统计特征：包括PMI、IDF、textrank值、前后词互信息、左右邻熵、
    独立检索占比（term单独作为query的qv/所有包含term的query的qv和）、统计概率TF、idf变种iqf
    """

    def __init__(
            self,
            ngram=4,
            pmi_path=pmi_path,
            entropy_path=entropy_path,
            segment_sep=' ',
            stopwords_path=default_stopwords_path,
    ):
        self.tfidf_model = TfIdf(stopwords_path=stopwords_path)
        self.textrank_model = TextRank()
        self.segment_sep = segment_sep
        self.pmi_model = PMI(
            ngram=ngram,
            pmi_path=pmi_path,
            entropy_path=entropy_path
        )

    def _get_tags_score(self, word, tags):
        score = 0.0
        for i, s in tags:
            if word == i:
                score = s
        return score

    @staticmethod
    def read_text(file_path, col_sep=',', limit_len=1000000):
        text = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split(col_sep)
                text += parts[0]
        if limit_len > 0:
            result = text[:limit_len]
        else:
            result = text
        return result

    def get_feature(self, sentence, is_word_segmented=False):
        """
        Get statistics feature
        :param sentence:
        :param is_word_segmented:
        :return: list, list: term features, sentence features
        """
        term_features = []
        sentence_features = {}
        rank_tags = self.textrank_model.extract(sentence)
        tfidf_tags = self.tfidf_model.extract(sentence)

        if is_word_segmented:
            word_seq = sentence.split(self.segment_sep)
        else:
            word_seq = word_segment(sentence, cut_type='word', pos=False)

        count = 0
        for word in word_seq:
            idf = self.tfidf_model.idf_freq.get(word, self.tfidf_model.median_idf)
            # PMI & Entropy
            left_word = word_seq[count - 1] if count > 0 else ''
            right_word = word_seq[count + 1] if count < len(word_seq) - 1 else ''
            left_right_word = ''.join([left_word, right_word])
            entropy_score = self.pmi_model.entropy_score(left_right_word)
            term_features.append(AttrDict(
                idf=idf,
                text_rank_score=self._get_tags_score(word, rank_tags),
                tfidf_score=self._get_tags_score(word, tfidf_tags),
                pmi_score=self.pmi_model.pmi_score(left_right_word),
                left_entropy_score=entropy_score[0],
                right_entropy_score=entropy_score[1],
            ))
            count += 1

        return term_features, sentence_features


class PMI:
    """
    前后词互信息PMI: 计算两个词语之间的关联程度，pmi越大越紧密
    左右邻熵：左右熵值越大，说明该词的周边词越丰富，意味着词的自由程度越大，其成为一个独立的词的可能性也就越大。
    """

    def __init__(
            self,
            text: str = None,
            ngram: int = 4,
            pmi_path: Optional[str] = pmi_path,
            entropy_path: Optional[str] = entropy_path
    ):
        if text is not None and len(text) > 0:
            logger.info(f'Use input text to generate new PMI dict, pmi path: {pmi_path}, entropy path: {entropy_path}')
            text = text.strip()
            words_freq = self.ngram_freq(text, ngram)
            self.pmi_score_dict = self.generate_pmi_score(words_freq, pmi_path)
            self.entropy_score_dict = self.generate_entropy_score(words_freq, text, entropy_path)
        else:
            self.pmi_score_dict = load_json(pmi_path)
            self.entropy_score_dict = load_json(entropy_path)
            logger.debug('Loaded PMI dict: %s' % pmi_path)

    @staticmethod
    def ngram_freq(text, ngram=4):
        """
        Get Ngram freq dict
        :param text: input text, sentence
        :param ngram: N
        :return: word frequency dict
        """
        words = []
        for i in range(1, ngram + 1):
            words += [text[j:j + i] for j in range(len(text) - i + 1)]
        words_freq = dict(Counter(words))
        return words_freq

    @staticmethod
    def generate_pmi_score(word_freq_dict, pmi_path=None):
        """
        Generate PMI score
        :param word_freq_dict: dict
        :return: dict(word, float), pmi score
        """
        result = dict()
        for word in word_freq_dict:
            if len(word) > 1:
                p_x_y = min([word_freq_dict.get(word[:i]) * word_freq_dict.get(word[i:]) for i in range(1, len(word))])
                score = p_x_y / word_freq_dict.get(word)
                result[word] = score
        if pmi_path:
            save_json(result, pmi_path)
            logger.info('Save pmi score to %s' % pmi_path)
        return result

    @staticmethod
    def entropy_list_score(char_list):
        """
        Get entropy score
        :param char_list: input char list
        :return: float, score
        """
        char_freq_dict = dict(Counter(char_list))
        char_size = len(char_list)
        entropy = -1 * sum([char_freq_dict.get(i) / char_size * np.log2(char_freq_dict.get(i) / char_size)
                            for i in char_freq_dict])
        return entropy

    def generate_entropy_score(self, word_freq_dict, text, entropy_path=None):
        """
        Generate entropy score
        :param word_freq_dict: dict
        :param text: input text, document
        :param entropy_path: save entropy file path
        :return: dict
        """
        result = dict()
        for word in word_freq_dict:
            if len(word) == 1:
                continue
            # pass error pattern
            if '*' in word:
                continue
            try:
                left_right_char = re.findall('(.)%s(.)' % word, text)
                left_char = [i[0] for i in left_right_char]
                left_entropy = self.entropy_list_score(left_char)

                right_char = [i[1] for i in left_right_char]
                right_entropy = self.entropy_list_score(right_char)
                if left_entropy > 0 or right_entropy > 0:
                    result[word] = [left_entropy, right_entropy]
            except Exception as e:
                logger.warning('error word %s, %s' % (word, e))
        if entropy_path:
            save_json(result, entropy_path)
            logger.info('Save entropy score to %s' % entropy_path)
        return result

    def pmi_score(self, word):
        """
        Get PMI score
        :param word:
        :return:
        """
        return self.pmi_score_dict.get(word, 0.0)

    def entropy_score(self, word):
        """
        Get entropy score
        :param word:
        :return:
        """
        return self.entropy_score_dict.get(word, [0.0, 0.0])


class TextFeature:
    """
    文本特征：包括Query长度、Term长度，Term在Query中的偏移量，term词性、长度信息、
    term数目、位置信息、句法依存tag、是否数字、是否英文、是否停用词、是否专名实体、
    是否重要行业词、embedding模长、删词差异度、以及短语生成树得到term权重等。
    """

    def __init__(
            self,
            stopwords_path=default_stopwords_path,
            person_name_path=person_name_path,
            place_name_path=place_name_path,
            common_char_path=common_char_path,
            segment_sep=' ',
    ):
        self.stopwords = self.load_set_file(stopwords_path)
        self.person_names = self.load_set_file(person_name_path)
        self.place_names = self.load_set_file(place_name_path)
        self.common_chars = self.load_set_file(common_char_path)
        self.segment_sep = segment_sep

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
        # logger.debug('%s' % word_seq)

        # sentence
        sentence_features = AttrDict(
            query_length=len(query),
            term_size=len(word_seq),
        )

        # term
        idx = 0
        offset = 0
        for word in word_seq:
            word_list = deepcopy(word_seq)
            if word in word_list:
                word_list.remove(word)
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
                is_common_char=self.is_common_char_string(word),
            ))
            idx += len(word)
            offset += 1

        return term_features, sentence_features


class NGram:
    def __init__(self, model_name_or_path=None, cache_folder=os.path.expanduser('~/.pycorrector/datasets/')):
        if model_name_or_path and os.path.exists(model_name_or_path):
            logger.info('Load kenlm language model:{}'.format(model_name_or_path))
            language_model_path = model_name_or_path
        else:
            # 语言模型 2.95GB
            get_file(
                'zh_giga.no_cna_cmn.prune01244.klm',
                'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
                extract=True,
                cache_subdir=cache_folder,
                verbose=1)
            language_model_path = os.path.join(cache_folder, 'zh_giga.no_cna_cmn.prune01244.klm')
        try:
            import kenlm
        except ImportError:
            raise ImportError('Kenlm not installed, use "pip install -U kenlm".')
        self.lm = kenlm.Model(language_model_path)
        logger.debug('Loaded language model: %s.' % language_model_path)

    def ngram_score(self, sentence: str):
        """
        取n元文法得分
        :param sentence: str, 输入的句子
        :return:
        """
        return self.lm.score(' '.join(sentence), bos=False, eos=False)

    def perplexity(self, sentence: str):
        """
        取语言模型困惑度得分，越小句子越通顺
        :param sentence: str, 输入的句子
        :return:
        """
        return self.lm.perplexity(' '.join(sentence))


class LanguageFeature:
    """
    语言模型特征：整个query的语言模型概率/去掉该Term后的Query的语言模型概率。
    """

    def __init__(self, segment_sep=' '):
        self.segment_sep = segment_sep
        self.ngram = NGram()

    def get_feature(self, query, is_word_segmented=False):
        """
        Get language feature
        :param query:
        :param is_word_segmented:
        :return: list, list: term features, sentence features
        """
        term_features = []
        if is_word_segmented:
            word_seq = query.split(self.segment_sep)
        else:
            word_seq = word_segment(query, cut_type='word', pos=False)

        # sentence
        sentence_features = AttrDict(
            ppl=self.ngram.perplexity(word_seq),
        )

        # term
        count = 0
        for word in word_seq:
            word_list = deepcopy(word_seq)
            if word in word_list:
                word_list.remove(word)
            left_word = word_seq[count - 1] if count > 0 else ''
            right_word = word_seq[count + 1] if count < len(word_seq) - 1 else ''

            term_features.append(AttrDict(
                del_term_ppl=self.ngram.perplexity(word_list),
                term_ngram_score=self.ngram.ngram_score(word),
                left_term_score=self.ngram.ngram_score(left_word + word),
                right_term_score=self.ngram.ngram_score(word + right_word)
            ))
            count += 1
        return term_features, sentence_features


class WordRank:
    """WordRank keyphrase extraction model."""

    def __init__(
            self,
            model_path=default_model_path,
            stopwords_path=default_stopwords_path,
            person_name_path=person_name_path,
            place_name_path=place_name_path,
            common_char_path=common_char_path,
            segment_sep=' ',
            ngram=4,
            pmi_path=pmi_path,
            entropy_path=entropy_path,
    ):
        """ init word rank model

        Args:
            model_path (str): the path to load the model in pickle format,
                default to "~/.cache/pke_zh/wordrank_model.pkl".
        """
        self.text_feature = TextFeature(
            stopwords_path=stopwords_path,
            person_name_path=person_name_path,
            place_name_path=place_name_path,
            common_char_path=common_char_path,
            segment_sep=segment_sep
        )
        self.statistics_feature = StatisticsFeature(
            ngram=ngram,
            pmi_path=pmi_path,
            entropy_path=entropy_path,
            segment_sep=segment_sep,
            stopwords_path=stopwords_path,
        )
        self.language_feature = LanguageFeature()
        self.segment_sep = segment_sep
        self.model = None
        if not os.path.exists(model_path):
            # release 的模型
            cache_folder = os.path.abspath(os.path.dirname(model_path))
            file_name = 'wordrank_model.pkl'
            get_file(
                f'{file_name}',
                f'https://github.com/shibing624/pke_zh/releases/download/0.2.2/{file_name}',
                extract=False,
                cache_subdir=cache_folder,
                verbose=1)
            model_path = os.path.join(cache_folder, f'{file_name}')
        if model_path and os.path.exists(model_path):
            self.model = load_pkl(model_path)
            logger.debug('Loaded wordrank model: {}'.format(model_path))
        self.model_path = model_path

    @staticmethod
    def data_reader(file_path, col_sep=','):
        """
        Load data
        :param file_path:
        :param col_sep:
        :return: list, list: contents, labels
        """
        contents = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                parts = line.split(col_sep)
                if len(parts) != 2:
                    continue
                contents.append(parts[0])
                labels.append(parts[1])
        return contents, labels

    def get_feature(self, query, is_word_segmented=False):
        """
        Get feature from query
        :param query: input query
        :param is_word_segmented: bool, is word segmented or not
        :return: features, terms
        """
        features = []
        terms = []

        text_terms, text_sent = self.text_feature.get_feature(query, is_word_segmented=is_word_segmented)
        stat_terms, stat_sent = self.statistics_feature.get_feature(query, is_word_segmented=is_word_segmented)
        lang_terms, lang_sent = self.language_feature.get_feature(query, is_word_segmented=is_word_segmented)
        # sentence feature
        text_sent.update(stat_sent)
        text_sent.update(lang_sent)
        # logger.debug('sentence features: %s' % text_sent)
        sent_feature = [text_sent.query_length, text_sent.term_size, text_sent.ppl]
        # term feature
        for text, stat, lang in zip(text_terms, stat_terms, lang_terms):
            text.update(stat)
            text.update(lang)
            # logger.debug('term features: %s' % text)
            term_feature = [
                text.term_length, text.idx, text.offset,
                float(text.is_number), float(text.is_chinese), float(text.is_alphabet),
                float(text.is_stopword), float(text.is_name), float(text.is_common_char),
                text.idf,
                text.text_rank_score, text.tfidf_score, text.pmi_score,
                text.left_entropy_score, text.right_entropy_score, text.del_term_ppl,
                text.term_ngram_score, text.left_term_score, text.right_term_score
            ]
            feature = sent_feature + term_feature
            features.append(feature)
            terms.append(text.term)
        return features, terms

    def train(self, train_file, col_sep=',', is_word_segmented=True):
        # 1.read train data
        contents, labels = self.data_reader(train_file, col_sep)
        logger.info('contents size:%s, labels size:%s' % (len(contents), len(labels)))

        features = []
        tags = []
        for content, label in zip(contents, labels):
            label_split = [int(i) for i in label.split(self.segment_sep)]
            content_split = content.split(self.segment_sep)
            if len(label_split) != len(content_split):
                logger.warning('pass, content size not equal label size, %s %s' % (content, label))
                continue
            tags += label_split
            content_features, terms = self.get_feature(content, is_word_segmented=is_word_segmented)
            features += content_features
        logger.info("[train]features size: %s, tags size: %s" % (len(features), len(tags)))
        assert len(features) == len(tags), "features size must equal tags size"
        X_train, X_val, y_train, y_val = train_test_split(features, tags, test_size=0.1, random_state=0)
        logger.debug("train size:%s, val size:%s" % (len(y_train), len(y_val)))
        # 3.train classification model, save model file
        model = RandomForestClassifier(n_estimators=300)
        # fit
        logger.debug("start train model ...")
        model.fit(X_train, y_train)
        # save model
        save_pkl(model, self.model_path, overwrite=True)
        logger.info("model saved: %s" % self.model_path)

        # 4.validation and evaluate
        logger.debug("evaluate model with validation data")
        self.evaluate(model, X_val, y_val)
        self.model = model
        return model

    def predict(self, query, is_word_segmented=False):
        logger.info('model predict')
        features, terms = self.get_feature(query, is_word_segmented=is_word_segmented)
        # predict classification model
        if self.model_path and os.path.exists(self.model_path):
            self.model = load_pkl(self.model_path)
            logger.debug('Loaded model: {}'.format(self.model_path))
        else:
            logger.error('model not found. path: {}'.format(self.model_path))
        logger.debug("model predict")
        label_pred = self.model.predict(features)
        logger.info("words: %s" % terms)
        logger.info("predict label: %s" % label_pred)
        return label_pred

    @staticmethod
    def evaluate(model, test_data, test_label, pred_save_path=None):
        print('{0}, val mean acc:{1}'.format(model.__str__(), model.score(test_data, test_label)))
        # multi
        label_pred = model.predict(test_data)
        # precision_recall_curve: multiclass format is not supported
        print(classification_report(test_label, label_pred))
        if pred_save_path:
            with open(pred_save_path, 'w', encoding='utf-8') as f:
                for i in label_pred:
                    f.write(str(i) + '\n')
        return label_pred

    def extract(self, input_string, n_best=10):
        """
        Extract keywords from text
        :param input_string:
        :param n_best:
        :return:
        """
        term_weights = []
        text = convert_to_unicode(input_string)
        if len(text) == 1:
            term_scores = zip([text], [0.0])
        else:
            # get feature
            data_feature, terms = self.get_feature(text, is_word_segmented=False)
            # predict model
            label_pred = self.model.predict(data_feature)
            term_scores = zip(terms, label_pred)
        for w, p in term_scores:
            if w not in term_weights:
                term_weights.append((w, p))

        keyphrases = sorted(term_weights, key=lambda k: k[1], reverse=True)[:n_best]
        return keyphrases
