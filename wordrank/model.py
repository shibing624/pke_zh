# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from wordrank.feature import Feature
from wordrank.utils.io_utils import save_pkl, load_pkl
from wordrank.utils.logger import logger


def data_reader(file_path, col_sep='\t'):
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
          ):
    # 1.read train data
    contents, labels = data_reader(train_file, col_sep)
    logger.info('contents size:%s, labels size:%s' % (len(contents), len(labels)))

    # 2.get feature
    feat = Feature(stopwords_path=stopwords_path,
                   person_name_path=person_name_path,
                   place_name_path=place_name_path,
                   common_char_path=common_char_path,
                   segment_sep=segment_sep,
                   domain_sample_path=domain_sample_path,
                   ngram=ngram,
                   pmi_path=pmi_path,
                   entropy_path=entropy_path)

    features = []
    tags = []
    for content, label in zip(contents, labels):
        label_split = label.split(segment_sep)
        content_split = content.split(segment_sep)
        if len(label_split) != len(content_split):
            logger.warning('pass, content size not equal label size, %s %s' % (content, label))
            continue
        tags += label_split
        content_features, terms = feat.get_feature(content, is_word_segmented=True)
        features += content_features
    logger.info("[train]features size: %s, tags size: %s" % (len(features), len(tags)))
    assert len(features) == len(tags), "features size must equal tags size"
    X_train, X_val, y_train, y_val = train_test_split(features, tags, test_size=0.2, random_state=0)
    logger.debug("train size:%s, val size:%s" % (len(y_train), len(y_val)))
    # 3.train classification model, save model file
    model = RandomForestClassifier(n_estimators=300)
    # fit
    logger.debug("start train model ...")
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
    # get feature
    feat = Feature(stopwords_path=stopwords_path,
                   person_name_path=person_name_path,
                   place_name_path=place_name_path,
                   common_char_path=common_char_path,
                   segment_sep=segment_sep,
                   domain_sample_path=domain_sample_path,
                   ngram=ngram,
                   pmi_path=pmi_path,
                   entropy_path=entropy_path
                   )
    features, terms = feat.get_feature(query, is_word_segmented=False)
    # predict classification model
    model = load_pkl(model_path)
    logger.debug("model predict")
    label_pred = model.predict(features)
    logger.info("terms: %s" % terms)
    logger.info("predict label: %s" % label_pred)
    print("predict label: %s" % label_pred)
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
