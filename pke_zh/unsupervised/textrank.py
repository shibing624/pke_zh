# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

TextRank keyphrase extraction model.

Implementation of the TextRank model for keyword extraction described in:

* Rada Mihalcea and Paul Tarau.
  TextRank: Bringing Order into Texts
  *In Proceedings of EMNLP*, 2004.

"""

import math
from loguru import logger
import networkx as nx

from pke_zh.base import BaseKeywordExtractModel


class TextRank(BaseKeywordExtractModel):
    """TextRank for keyword extraction.

    This model builds a graph that represents the text. A graph based ranking
    algorithm is then applied to extract the lexical units (here the words) that
    are most important in the text.

    In this implementation, nodes are words of certain part-of-speech (nouns
    and adjectives) and edges represent co-occurrence relation, controlled by
    the distance between word occurrences (here a window of 2 words). Nodes
    are ranked by the TextRank graph-based ranking algorithm in its unweighted
    variant.
    """

    def __init__(self):
        """Redefining initializer for TextRank."""
        super(TextRank, self).__init__()

        self.graph = nx.Graph()
        """The word graph."""

    def clear_cache(self):
        self.graph.clear()

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.
        :param pos: set of valid POS tags, defaults to ('NOUN', 'PROPN',
            'ADJ').
        """
        if pos is None:
            pos = {'n', 'a'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

    def calculate_bi_nodes_weights(self, wi, window, text, node1):
        for wj in range(wi + 1, min(wi + window, len(text))):
            node2, is_in_graph2 = text[wj]
            if is_in_graph2 and node1 != node2:
                self.graph.add_edge(node1, node2)

    def _flatten_words_and_add_nodes(self, pos):
        text = [(word, sentence.pos[i] in pos) for sentence in self.sentences
                for i, word in enumerate(sentence.words)]

        # add nodes to the graph
        self.graph.add_nodes_from([word for word, valid in text if valid])

        return text

    def build_word_graph(self, window=2, pos=None):
        """Build a word co-occurrence graph representation of the document

        Syntactic filters can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance between
        word occurrences in the document.

        As the original paper does not give precise details on how the word
        graph is constructed, we make the following assumptions from the example
        given in Figure 2: 1) sentence boundaries **are not** taken into account
        and, 2) stopwords and punctuation marks **are** considered as words when
        computing the window.

        :param window: int, the window for connecting two words in the graph,
            defaults to 2.
        :param pos: the set of valid pos for words to be considered as nodes
            in the graph, defaults to ('NOUN'', 'ADJ').
        """
        if pos is None:
            pos = {'n', 'a'}
        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = self._flatten_words_and_add_nodes(pos)

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):
            # speed up things
            if not is_in_graph1:
                continue
            self.calculate_bi_nodes_weights(i, window, text, node1)

    def candidate_weighting(self,
                            window=2,
                            pos=None,
                            top_percent=None,
                            normalized=False):
        """Tailored candidate ranking method for TextRank.

        Keyphrase candidates are either composed from the T-percent highest-ranked words as in the
        original paper or extracted using the `candidate_selection()` method.
        Candidates are ranked using the sum of their (normalized?) words.

        :param window: int, the window for connecting two words in the graph,
            defaults to 2.
        :param pos: the set of valid pos for words to be considered as nodes
            in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        :param top_percent: float, percentage of top vertices to keep for phrase
            generation.
        :param normalized: bool, normalize keyphrase score by their length,
            defaults to False.
        """
        if pos is None:
            pos = {'n', 'a'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # compute the word scores using the unweighted PageRank formulae
        word_scores = nx.pagerank_scipy(self.graph, alpha=0.85, tol=0.0001, weight=None)

        # generate the phrases from the T-percent top ranked words
        if top_percent is not None:
            # computing the number of top keywords
            nb_nodes = self.graph.number_of_nodes()
            to_keep = min(math.floor(nb_nodes * top_percent), nb_nodes)

            # sorting the nodes by decreasing scores
            top_words = sorted(word_scores, key=word_scores.get, reverse=True)

            # creating keyphrases from the T-top words
            self.longest_keyword_sequence_selection(top_words[:int(to_keep)])
        # weight candidates using the sum of their word scores
        self.candidate_weight_calculate(word_scores, normalized)

    def candidate_weight_calculate(self, word_scores, normalized):
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([word_scores[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)
            # use position to break ties
            self.weights[k] += (self.candidates[k].offsets[0] * 1e-8)

    def extract(self, input_file_or_string, n_best=10, pos=None, top_percent=0.33):
        self.load_document(input=input_file_or_string,
                           language='zh',
                           normalization=None)

        # clear previous graph is exists
        self.clear_cache()

        self.candidate_weighting(
            window=2,
            pos=pos,
            top_percent=top_percent
        )
        keyphrases = self.get_n_best(n=n_best)
        return keyphrases
