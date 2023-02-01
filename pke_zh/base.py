# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import re
import os
from string import punctuation
from loguru import logger
import codecs
from collections import defaultdict
from six import string_types

from pke_zh.data_structures import Candidate, Document
from pke_zh.readers import RawTextReader

pwd_path = os.path.abspath(os.path.dirname(__file__))
# inner data file
default_stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')

map_tag = {'ad': 'a', 'ag': 'a', 'an': 'a',
           'ng': 'n', 'nr': 'n', 'nrfg': 'n', 'nrt': 'n', 'ns': 'n', 'nt': 'n', 'nz': 'n',
           'rg': 'r', 'rr': 'r', 'rz': 'r',
           }


def load_stopwords(file_path):
    stopwords = set()
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                stopwords.add(line)
    return stopwords


class BaseKeywordExtractModel(object):
    """The ket class that provides base functions."""

    def __init__(self, stopwords_path='', valid_pos=None, self_defined_keyword_path=None):
        """Initializer for base class."""

        self.input_file = None
        """Path to the input file."""

        self.language = None
        """Language of the input file."""

        self.normalization = None
        """Word normalization method."""

        self.sentences = []
        """Sentence container (list of Sentence objects)."""

        self.candidates = defaultdict(Candidate)
        """Keyphrase candidates container (dict of Candidate objects)."""

        self.weights = {}
        """Weight container (can be either word or candidate weights)."""

        self.stoplist = list(load_stopwords(stopwords_path)) if stopwords_path and os.path.exists(stopwords_path) \
            else list(load_stopwords(default_stopwords_path))
        """List of stopwords."""

        self.valid_pos = {'n', 'a'} if valid_pos is None else valid_pos

        punctuation_expand = ['/', ',', '$', '%', '^', '*', '(', '+', '"', "'", ']', '+', '|', '[', '+', '——', '！',
                              '，', '、', '~', '@', '#', '￥', '%', '&', '*', '（', '）', '：', '；', '《', '）', '《',
                              '》', '“', '”', '(', ')', '»', '〔', '〕', '-']
        self.punctuations = list(punctuation) + punctuation_expand

        self.self_defined_keywords = []
        if self_defined_keyword_path is not None:
            words = open(self_defined_keyword_path, 'r', encoding='utf-8').readlines()
            words = [x.strip for x in words if x.strip()]
            words = sorted(words, key=lambda w: len(w), reverse=True)
            if len(words) > 10000:
                logger.warning(
                    "Too much matching-keyword! We recommend you try the HugeKeywordMatching algo.")
            self.self_defined_keywords.extend(words)

        self.raw_text = ""

    def clear_cache(self):
        raise NotImplementedError

    def load_document(self, input, **kwargs):
        """Loads the content of a document/string/stream in a given language.

        :param input: str, input.
        :param language: str, language of the input, defaults to 'en'.
        :param encoding: str, encoding of the raw file.
        :param normalization: str, word normalization method, defaults to
            'stemming'. Other possible values are 'lemmatization' or 'None'
            for using word surface forms instead of stems/lemmas.
        """
        # get the language parameter
        language = kwargs.get('language', 'zh')

        # test whether the language is known, otherwise fall back to english
        kwargs['language'] = language

        # initialize document
        doc = Document()
        self.raw_text = ""

        if isinstance(input, string_types):
            # if input is an input file
            if os.path.isfile(input):
                # other extensions are considered as raw text
                parser = RawTextReader(language=language)
                encoding = kwargs.get('encoding', 'utf-8')
                with codecs.open(input, 'r', encoding=encoding) as file:
                    self.raw_text = file.read()
                doc = parser.read(text=self.raw_text, path=input, **kwargs)

            # if input is a string
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input, **kwargs)
                self.raw_text = input
        # set the input file
        self.input_file = doc.input_file

        # set the language of the document
        self.language = language

        # set the sentences
        self.sentences = doc.sentences

        # POS normalization
        self.normalize_pos_tags()

        # clear previous candidates and weights if exists
        self.candidates.clear()
        self.weights.clear()

    def normalize_pos_tags(self):
        """Normalizes some jieba's postags."""
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].pos = [map_tag.get(tag, tag) for tag in sentence.pos]

    @staticmethod
    def is_redundant(candidate, prev, minimum_length=1):
        """Test if one candidate is redundant

        A candidate is considered redundant if it is
        included in another candidate that is ranked higher in the list.

        :param candidate: str, the lexical form of the candidate.
        :param prev: list, the list of already selected candidates (lexical forms).
        :param minimum_length: int, minimum length (in words) of the candidate
            to be considered, defaults to 1.
        """
        # get the tokenized lexical form from the candidate
        candidate = re.sub(r'\s', '', candidate)

        # only consider candidate greater than one word
        if len(candidate) < minimum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [re.sub(r'\s', '', u) for u in prev]

        # loop through the already selected candidates
        redundant_status = any([prev_candidate in candidate or candidate in prev_candidate
                                for prev_candidate in prev])

        return redundant_status

    def redundancy_removal_best(self, best, num_candidates):
        # initialize a new container for non redundant candidates
        non_redundant_best = []

        # loop through the best candidates
        for candidate in best:
            # test weather candidate is redundant
            if self.is_redundant(candidate, non_redundant_best):
                continue

            # add the candidate otherwise
            non_redundant_best.append(candidate)

            # break computation if the n-best are found
            if len(non_redundant_best) >= num_candidates:
                break

        # copy non redundant candidates in best container
        return non_redundant_best

    def n_best_surface_form(self, stemming, best, top_n, n_best):
        if not stemming and isinstance(self.candidates, dict):
            n_best = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                       self.weights[u]) for u in best[:min(top_n, len(best))]]
        return n_best

    def get_n_best(self, n=10, redundancy_removal=True, stemming=False):
        """Returns the n-best candidates given the weights.

        :param n: int, the number of candidates, defaults to 10.
        :param redundancy_removal: bool, whether redundant keyphrases are
            filtered out from the n-best list, defaults to False.
        :param stemming: bool, whether to extract stems or surface forms
            (lowercased, first occurring form of candidate), default to
            False.
        """
        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)

        # remove redundant candidates
        if redundancy_removal:
            best = self.redundancy_removal_best(best, n)

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        # when used in KPRank, self.candidates is a list.
        n_best = self.n_best_surface_form(stemming, best, n, n_best)

        # remove blank space of keyphrases
        n_best = [(re.sub(r'\s', '', u), weight) for u, weight in n_best]

        # if len(n_best) < n:
        #     logger.debug(
        #         'Not enough candidates to choose from '
        #         '({} requested, {} given)'.format(n, len(n_best)))

        # return the list of best candidates
        return n_best

    def add_candidate(self, words, pos, offset, sentence_id):
        """Add a keyphrase candidate to the candidates container.

        :param words: list, the words (surface form) of the candidate.
        :param pos: list, the Part-Of-Speeches of the words in the candidate.
        :param offset: int, the offset of the first word of the candidate.
        :param sentence_id: int, the sentence id of the candidate.
        """
        # build the lexical (canonical) form of the candidate using stems
        lexical_form = ' '.join(words)

        # add/update the surface forms
        self.candidates[lexical_form].surface_forms.append(words)

        # add/update the lexical_form
        self.candidates[lexical_form].lexical_form = words

        # add/update the POS patterns
        self.candidates[lexical_form].pos_patterns.append(pos)

        # add/update the offsets
        self.candidates[lexical_form].offsets.append(offset)

        # add/update the sentence ids
        self.candidates[lexical_form].sentence_ids.append(sentence_id)

    def ngram_selection(self, n=3):
        """Select all the n-grams and populate the candidate container.

        :param n: int, the n-gram length, defaults to 3.
        """
        # loop through the sentences
        for i, sentence in enumerate(self.sentences):
            # limit the maximum n for short sentence
            skip = min(n, sentence.length)
            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])
            # generate the ngrams
            for j in range(sentence.length):
                for k in range(j + 1, min(j + 1 + skip, sentence.length + 1)):
                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[j:k],
                                       # stems=sentence.stems[j:k],
                                       pos=sentence.pos[j:k],
                                       offset=shift + j,
                                       sentence_id=i)

    def longest_pos_sequence_selection(self, valid_pos=None):
        self.longest_sequence_selection(
            key=lambda s: s.pos, valid_values=valid_pos)

    def longest_keyword_sequence_selection(self, keywords):
        self.longest_sequence_selection(
            key=lambda s: s.words, valid_values=keywords)

    def longest_sequence_selection(self, key, valid_values):
        """Select the longest sequences of given POS tags as candidates.

        :param key: function that given a sentence return an iterable
        :param valid_values: set, the set of valid values, defaults to None.
        """
        # loop through the sentences
        for i, sentence in enumerate(self.sentences):
            # compute the offset shift for the sentence
            shift = sum(list(map(lambda s: s.length, self.sentences[0:i])))
            # container for the sequence (defined as list of offsets)
            seq = []

            # loop through the tokens
            for j, value in enumerate(key(self.sentences[i])):
                # add candidate offset in sequence and continue if not last word
                if value in valid_values:
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue

                # add sequence as candidate if non empty
                self.add_seq_as_candidates(seq, sentence, shift, i, j)

                # flush sequence container
                seq = []

    def add_seq_as_candidates(self, seq, sentence, shift, i, j):
        if len(seq) == 0:
            return

        # bias for candidate in last position within sentence
        bias = 0
        if j == (sentence.length - 1):
            bias = 1

        # add the ngram to the candidate container
        self.add_candidate(words=sentence.words[seq[0]:seq[-1] + 1],
                           pos=sentence.pos[seq[0]:seq[-1] + 1],
                           offset=shift + j - len(seq) + bias,
                           sentence_id=i)

    @staticmethod
    def _is_alphanum(word, valid_punctuation_marks='-'):
        """Check if a word is valid, i.e. it contains only alpha-numeric
        characters and valid punctuation marks.

        :param word: str, a word.
        :param valid_punctuation_marks: str, punctuation marks that are valid
                for a candidate, defaults to '-'.
        """
        for punct in valid_punctuation_marks.split():
            word = word.replace(punct, '')
        return word.isalnum()

    @staticmethod
    def discard_length_invalid_candidate(k, v, words, minimum_length, minimum_word_size, maximum_word_number):
        # discard candidates composed of 1-2 characters
        if len(''.join(words)) < minimum_length:
            return k

        # discard candidates containing small words (1-character)
        elif min([len(u) for u in words]) < minimum_word_size:
            return k

        # discard candidates composed of more than 5 words
        elif len(v.lexical_form) > maximum_word_number:
            return k
        return ''

    def discard_candidate(self, k, v, stoplist=None, minimum_length=2, minimum_word_size=2, valid_punctuation_marks='-',
                          maximum_word_number=7, pos_blacklist=None):
        # get the words from the first occurring surface form
        words = [u.lower() for u in v.surface_forms[0]]

        # discard if words are in the stoplist
        if stoplist is None:
            stoplist = set()
        if set(words).intersection(stoplist):
            return k
        # discard if tags are in the pos_blacklist
        elif set(v.pos_patterns[0]).intersection(pos_blacklist):
            return k
        # discard if containing tokens composed of only punctuation
        elif any([set(u).issubset(set(self.punctuations)) for u in words]):
            return k
        r = self.discard_length_invalid_candidate(k, v, words, minimum_length, minimum_word_size,
                                                  maximum_word_number)
        return r

    def candidate_filtering(self,
                            stoplist=None,
                            minimum_length=2,
                            minimum_word_size=2,
                            valid_punctuation_marks='-',
                            maximum_word_number=7,
                            pos_blacklist=None):
        """Filter the candidates containing strings from the stoplist.

        Only keep the candidates containing alpha-numeric characters (if the
        non_latin_filter is set to True) and those length exceeds a given
        number of characters.

        :param stoplist: list of strings, defaults to None.
        :param minimum_length: int, minimum number of characters for a
            candidate, defaults to 3.
        :param minimum_word_size: int, minimum number of characters for a
            token to be considered as a valid word, defaults to 2.
        :param valid_punctuation_marks: str, punctuation marks that are valid
            for a candidate, defaults to '-'.
        :param maximum_word_number: int, maximum length in words of the
            candidate, defaults to 5.
        :param pos_blacklist: list of unwanted Part-Of-Speeches in
            candidates, defaults to [].
        """
        if pos_blacklist is None:
            pos_blacklist = []

        del_keys = set([self.discard_candidate(k, self.candidates[k], stoplist, minimum_length, minimum_word_size,
                                               valid_punctuation_marks, maximum_word_number, pos_blacklist) for k in
                        list(self.candidates)])
        for k in list(self.candidates):
            if k in del_keys:
                del self.candidates[k]

    def self_defined_keyword_matching(self):
        """

        :param docs: list of doc texts to be extracted
        :param keywords: self-defined keyword set
        :return:
        """
        result = []
        for w in self.self_defined_keywords:
            if w in self.raw_text:
                result.append(w)
        return result

    def extract(self, input_file_or_string, n_best=10, **kwargs):
        raise NotImplementedError
