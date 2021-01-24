# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from wordrank.features.language_feature import LanguageFeature
from wordrank.features.statistics_feature import StatisticsFeature
from wordrank.features.text_feature import TextFeature
from wordrank.utils.io_utils import save_pkl, load_pkl, build_vocab, load_vocab, write_vocab
from wordrank.utils.logger import logger


def data_reader(file_path, col_sep='\t'):
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


def tfidf_word_feature(data_set, is_infer=False, feature_vec_path='', word_vocab=None):
    """
    Get TFIDF ngram feature by word
    """
    if is_infer:
        vectorizer = load_pkl(feature_vec_path)
        data_feature = vectorizer.transform(data_set)
    else:
        vectorizer = TfidfVectorizer(analyzer='word', vocabulary=word_vocab, sublinear_tf=True)
        data_feature = vectorizer.fit_transform(data_set)
    vocab = vectorizer.vocabulary_
    logger.debug('vocab size: %d' % len(vocab))
    logger.debug(data_feature.shape)
    # if not self.is_infer:
    save_pkl(vectorizer, feature_vec_path, overwrite=True)
    return data_feature


def train(train_file,
          col_sep,
          stopwords_path,
          person_name_path,
          place_name_path,
          common_char_path,
          segment_sep,
          domain_sample_path,
          ngram,
          pmi_path,
          entropy_path,
          model_path,
          word_vocab_path,
          feature_vec_path,
          ):
    # 1.read train data
    contents, labels = data_reader(train_file, col_sep)
    logger.info('contents size:%s, labels size:%s' % (len(contents), len(labels)))
    word_lst = []
    for i in contents:
        word_lst.extend(i.split())

    # word vocab
    word_vocab = build_vocab(word_lst, min_count=1, sort=True, lower=True)
    # save word vocab
    write_vocab(word_vocab, word_vocab_path)
    word_id = load_vocab(word_vocab_path)

    # 2.get feature
    text_feature = TextFeature(
        stopwords_path=stopwords_path,
        person_name_path=person_name_path,
        place_name_path=place_name_path,
        common_char_path=common_char_path,
        segment_sep=segment_sep
    )
    statistics_feature = StatisticsFeature(
        domain_sample_path=domain_sample_path,
        ngram=ngram,
        pmi_path=pmi_path,
        entropy_path=entropy_path,
        segment_sep=segment_sep
    )
    language_feature = LanguageFeature(segment_sep=segment_sep)

    text_contents = []
    features = []
    tags = []
    for content, label in zip(contents, labels):
        text_terms, text_sent = text_feature.get_feature(content, is_word_segmented=True)
        stat_terms, stat_sent = statistics_feature.get_feature(content, is_word_segmented=True)
        lang_terms, lang_sent = language_feature.get_feature(content, is_word_segmented=True)
        # sentence feature
        text_sent.update(stat_sent)
        text_sent.update(lang_sent)
        logger.debug('sentence features: %s' % text_sent)
        sent_feature = [text_sent.query_length, text_sent.term_size, text_sent.ppl]

        label_split = label.split(segment_sep)
        if len(label_split) != text_sent.term_size:
            logger.warning('pass, content size not equal label size, %s %s' % (content, label))
            continue
        tags += label_split
        # term feature
        for text, stat, lang in zip(text_terms, stat_terms, lang_terms):
            text.update(stat)
            text.update(lang)
            # logger.debug('term features: %s' % text)
            term_feature = [text.term_length, text.idx, text.offset, float(text.is_number),
                            float(text.is_chinese), float(text.is_alphabet), float(text.is_stopword),
                            float(text.is_name), float(text.is_common_char), text.embedding_sum, text.del_term_score,
                            text.idf, text.text_rank_score, text.tfidf_score, text.pmi_score, text.left_entropy_score,
                            text.right_entropy_score, text.del_term_ppl, text.term_ngram_score, text.left_term_score,
                            text.right_term_score]
            feature = sent_feature + term_feature
            features.append(feature)

            text_contents.append(content + segment_sep + text.term)
    logger.info("features size: %s, tags size: %s" % (len(features), len(tags)))
    assert len(features) == len(tags), "features size must equal tags size"

    # add word tfidf feature
    machine_feature = tfidf_word_feature(text_contents, is_infer=False, feature_vec_path=feature_vec_path,
                                         word_vocab=word_id)
    # merge featrue
    human_feature = np.array(features, dtype=float)
    data_features = hstack([machine_feature, csr_matrix(human_feature)], 'csr')

    X_train, X_val, y_train, y_val = train_test_split(machine_feature, tags, test_size=0.2, random_state=0)
    logger.debug("train size:%s, val size:%s" % (len(y_train), len(y_val)))
    # 3.train classification model, save model file
    model = RandomForestClassifier(n_estimators=100)
    # fit
    logger.info("start train model ...")
    model.fit(X_train, y_train)
    # save model
    save_pkl(model, model_path, overwrite=True)
    logger.info("model saved: %s" % model_path)

    # 4.validation and evaluate
    logger.debug("evaluate model with validation data")
    evaluate(model, X_val, y_val)
    return model


def predict(query,
            model_path,
            stopwords_path,
            person_name_path,
            place_name_path,
            common_char_path,
            segment_sep,
            domain_sample_path,
            ngram,
            pmi_path,
            entropy_path,
            ):
    logger.info('model predict')
    # 2.get feature
    text_feature = TextFeature(
        stopwords_path=stopwords_path,
        person_name_path=person_name_path,
        place_name_path=place_name_path,
        common_char_path=common_char_path,
        segment_sep=segment_sep
    )
    statistics_feature = StatisticsFeature(
        domain_sample_path=domain_sample_path,
        ngram=ngram,
        pmi_path=pmi_path,
        entropy_path=entropy_path,
        segment_sep=segment_sep
    )
    language_feature = LanguageFeature(segment_sep=segment_sep)

    features = []
    text_terms, text_sent = text_feature.get_feature(query, is_word_segmented=False)
    stat_terms, stat_sent = statistics_feature.get_feature(query, is_word_segmented=False)
    lang_terms, lang_sent = language_feature.get_feature(query, is_word_segmented=False)
    # sentence feature
    text_sent.update(stat_sent)
    text_sent.update(lang_sent)
    logger.debug('sentence features: %s' % text_sent)
    sent_feature = [text_sent.query_length, text_sent.term_size, text_sent.ppl]
    # term feature
    for text, stat, lang in zip(text_terms, stat_terms, lang_terms):
        text.update(stat)
        text.update(lang)
        logger.debug('term features: %s' % text)
        term_feature = [text.term_length, text.idx, text.offset, float(text.is_number),
                        float(text.is_chinese), float(text.is_alphabet), float(text.is_stopword),
                        float(text.is_name), float(text.is_common_char), text.embedding_sum, text.del_term_score,
                        text.idf, text.text_rank_score, text.tfidf_score, text.pmi_score, text.left_entropy_score,
                        text.right_entropy_score, text.del_term_ppl, text.term_ngram_score, text.left_term_score,
                        text.right_term_score]
        feature = sent_feature + term_feature
        features.append(feature)
    logger.info("features size: %s" % len(features))
    data_feature = np.array(features, dtype=float)
    # 3.predict classification model
    model = load_pkl(model_path)
    logger.debug("model predict")
    label_pred = model.predict(data_feature)
    logger.info("predict label: %s" % label_pred)
    return label_pred


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
