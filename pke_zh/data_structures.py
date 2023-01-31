# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Data structures
modify from: https://github.com/boudinfl/pke/blob/master/pke/data_structures.py
"""


class Sentence(object):
    """The sentence data structure."""

    def __init__(self, words):

        self.words = words
        """list of words (tokens) in the sentence."""

        self.pos = []
        """list of Part-Of-Speeches."""

        self.length = len(words)
        """length (number of tokens) of the sentence."""

        self.meta = {}
        """meta-information of the sentence."""

    @staticmethod
    def _is_length_word_pos_meta_not_equal(length1, words1, pos1, meta1, length2, words2, pos2, meta2):
        if length1 != length2 or \
                words1 != words2 or \
                pos1 != pos2 or \
                meta1 != meta2:
            return True
        return False

    def __eq__(self, other):
        """Compares two sentences for equality."""

        # 1. test whether they are instances of different classes
        # 2. test whether they are of same length
        # 3. test whether they have the same words
        # 4. test whether they have the same PoS tags
        # 5. test whether they have the same meta-information
        if type(self) != type(other) or \
                self._is_length_word_pos_meta_not_equal(
                    self.length, self.words, self.pos, self.meta,
                    other.length, other.words, other.pos, other.meta
                ):
            # self.length != other.length or \
            # self.words != other.words or \
            # self.pos != other.pos or \
            # self.meta != other.meta:
            return False

        # if everything is ok then they are equal
        return True


class Candidate(object):
    """The keyphrase candidate data structure."""

    def __init__(self):
        self.surface_forms = []
        """ the surface forms of the candidate. """

        self.offsets = []
        """ the offsets of the surface forms. """

        self.sentence_ids = []
        """ the sentence id of each surface form. """

        self.pos_patterns = []
        """ the Part-Of-Speech patterns of the candidate. """

        self.lexical_form = []
        """ the lexical form of the candidate. """


def parse_sentence(sent_obj):
    s = Sentence(words=sent_obj['words'])

    # add the POS
    s.pos = sent_obj['POS']

    # add the meta-information, for chinese, it's `char_offset`
    m_key = "char_offsets"
    s.meta[m_key] = sent_obj[m_key]

    return s


class Document(object):
    """The Document data structure."""

    def __init__(self):

        self.input_file = None
        """ The path of the input file. """

        self.sentences = []
        """ The sentence container (list of Sentence). """

    @staticmethod
    def from_sentences(sentences, **kwargs):
        """Populate the sentence list.

        Args:
            sentences (Sentence list): content to create the document.
            input_file (str): path to the input file.
        """

        # initialize document
        doc = Document()

        # set the input file
        doc.input_file = kwargs.get('input_file', None)

        # loop the parsed sentences
        doc.sentences = [parse_sentence(sent) for sent in sentences]

        return doc

    def __eq__(self, other):
        """Compares two documents for equality."""

        # test whether they are instances of different classes
        if type(self) != type(other):
            return False

        # test whether they have the same input path
        if self.input_file != other.input_file:
            return False

        # test whether they contain the same lists of sentences
        if self.sentences != other.sentences:
            return False

        # if everything is ok then they are equal
        return True
