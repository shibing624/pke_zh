# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

PositionRank keyphrase extraction model.

PositionRank is an unsupervised model for keyphrase extraction from scholarly
documents that incorporates information from all positions of a word's
occurrences into a biased PageRank. The model is described in:

* Corina Florescu and Cornelia Caragea.
  PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly
  Documents.
  *In proceedings of ACL*, pages 1105-1115, 2017.
"""

from collections import defaultdict
import networkx as nx

from pke_zh.unsupervised.singlerank import SingleRank


class PositionRank(SingleRank):
    """PositionRank keyphrase extraction model"""

    def __init__(self):
        """Redefining initializer for PositionRank."""

        super(PositionRank, self).__init__()

        self.positions = defaultdict(float)
        """Container the sums of word's inverse positions."""

    def clear_cache(self):
        # clear previous cache
        self.positions.clear()
        self.graph.clear()

    def candidate_selection(
            self,
            pos=None,
            maximum_word_number=3,
            **kwargs
    ):
        """Candidate selection heuristic using a syntactic PoS pattern for noun phrase extraction.

        :param pos: valid postag set
        :param maximum_word_number:the maximum number of words allowed for
                keyphrase candidates, defaults to 3.
        :param kwargs:
        :return:
        """
        pos = {'n', 'a'}
        self.longest_pos_sequence_selection(valid_pos=pos)

        # filter candidates greater than 3 words
        for k in list(self.candidates):
            v = self.candidates[k]
            if len(v.lexical_form) > maximum_word_number:
                del self.candidates[k]

    @staticmethod
    def is_valid_edge(idx2, text_length, position1, text, window):
        return idx2 < text_length and (text[idx2][1] - position1) < window

    def add_graph_edges(self, text, window):
        for i, (node1, position1) in enumerate(text):
            j = i + 1
            while self.is_valid_edge(j, len(text), position1, text, window):
                node2, _ = text[j]
                if node1 != node2:
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, weight=0)
                    self.graph[node1][node2]['weight'] += 1
                j = j + 1

    def compute_word_reverse_position(self, text):
        for word, position in text:
            self.positions[word] += 1 / (position + 1)

    @staticmethod
    def filter_word_pos(sentence, pos, shift):
        valid_text = []
        for j, word in enumerate(sentence.words):
            if sentence.pos[j] in pos:
                valid_text.append((word, shift + j))
        return valid_text

    def build_word_graph(self, window=10, pos=None):
        """Build the graph representation of the document.

        In the graph, nodes are words that passes a Part-of-Speech filter. Two
        nodes are connected if the words corresponding to these nodes co-occur
        within a `window` of contiguous tokens. The weight of an edge is
        computed based on the co-occurrence count of the two words within a
        `window` of successive tokens.

        :param window: int, the window within the sentence for connecting two
            words in the graph, defaults to 10.
        :param pos: set, the set of valid pos for words to be considered as nodes
            in the graph, defaults to ('NOUN', 'PROP', 'ADJ').
        """
        if pos is None:
            pos = {'n', 'a'}

        # flatten document as a sequence of only valid (word, position) tuples
        text = []
        for i, sentence in enumerate(self.sentences):
            shift = sum([s.length for s in self.sentences[0:i]])
            valid_word_shifts = self.filter_word_pos(sentence, pos, shift)
            text.extend(valid_word_shifts)
        # add nodes to the graph
        self.graph.add_nodes_from([word for (word, position) in text])

        # add edges to the graph
        self.add_graph_edges(text, window)

        # compute the sums of the word's inverse positions
        self.compute_word_reverse_position(text)

    def candidate_weight_norm(self, pagerank_weight, normalized):
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([pagerank_weight.get(t, 0.0) for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)

    def candidate_weighting(self, window=10, pos=None, normalized=False):
        """Candidate weight calculation using a biased PageRank.

        :param window: int, the window within the sentence for connecting two
                words in the graph, defaults to 10.
        :param pos: the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        :param normalized: bool, normalize keyphrase score by their length,
                defaults to False.
        """
        if pos is None:
            pos = {'n', 'a'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)
        # normalize cumulated inverse positions
        norm = sum(self.positions.values())
        for word in self.positions:
            self.positions[word] /= norm

        # compute the word scores using biased random walk
        w = nx.pagerank(G=self.graph,
                        alpha=0.85,
                        tol=0.0001,
                        personalization=self.positions,
                        weight='weight')

        # loop through the candidates
        self.candidate_weight_norm(w, normalized)

    def extract(self, input_file_or_string, n_best=10, pos=None):
        # 1. valid postags
        if pos is None:
            pos = {'n', 'a'}
        # 2. clear cache
        self.clear_cache()
        # 3. load the content of the document.
        self.load_document(input=input_file_or_string, language='zh', normalization=None)
        # 4. select the noun phrases up to 3 words as keyphrase candidates.
        self.candidate_selection(pos=pos, maximum_word_number=3)
        self.candidate_weighting(window=10, pos=pos)
        keyphrases = self.get_n_best(n=n_best)
        return keyphrases
