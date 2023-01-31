# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

SingleRank keyphrase extraction model.

Simple extension of the TextRank model described in:

* Xiaojun Wan and Jianguo Xiao.
  CollabRank: Towards a Collaborative Approach to Single-Document Keyphrase
  Extraction.
  *In proceedings of the COLING*, pages 969-976, 2008.
"""
import networkx as nx
from pke_zh.unsupervised.textrank import TextRank


class SingleRank(TextRank):
    """SingleRank keyphrase extraction model.

    This model is an extension of the TextRank model that uses the number of
    co-occurrences to weigh edges in the graph.
    """

    def __init__(self):
        """Redefining initializer for SingleRank."""
        super(SingleRank, self).__init__()

    def calculate_bi_nodes_weights(self, word_idx, window, text, node1):
        for j in range(word_idx + 1, min(word_idx + window, len(text))):
            node2, is_in_graph2 = text[j]
            if is_in_graph2 and node1 != node2:
                if not self.graph.has_edge(node1, node2):
                    self.graph.add_edge(node1, node2, weight=0.0)
                self.graph[node1][node2]['weight'] += 1.0

    def extract_word_and_pos_from_sentences(self, valid_pos):
        text = [(word, sentence.pos[i] in valid_pos) for sentence in self.sentences
                for i, word in enumerate(sentence.words)]
        return text

    def build_word_graph(self, window=10, pos=None):
        """Build a co-occurrence word graph representation of the document

        Syntactic filters can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance (window) between word
        occurrences in the document. The number of times two words co-occur in a window is
        encoded as *edge weights*. Sentence boundaries **are not** taken into account in the window.

        :param window: int, the window for connecting two words in the graph,
            defaults to 10.
        :param pos: the set of valid pos for words to be considered as nodes
            in the graph, defaults to ('NOUN', 'ADJ').
        """
        if pos is None:
            pos = {'n', 'a'}

        # flatten document as a sequence of (word, pass_syntactic_filter) tuples
        text = self.extract_word_and_pos_from_sentences(pos)

        self.clear_cache()

        # add nodes to the graph
        valid_text = list(filter(lambda x: x[1], text))
        self.graph.add_nodes_from([word for word, _ in valid_text])
        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):
            # speed up things
            if not is_in_graph1:
                continue

            self.calculate_bi_nodes_weights(i, window, text, node1)

    def candidate_weighting(self, window=10, pos=None, top_percent=None, normalized=False):
        """Keyphrase candidate ranking using the weighted variant of the TextRank formulae.

        Candidates are scored by the sum of the scores of their words.

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

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph,
                              alpha=0.85,
                              tol=0.0001,
                              weight='weight')

        # loop through the candidates
        self.candidate_weight_calculate(w, normalized)

    def extract(self, input_file_or_string, n_best=10, pos=None):
        self.load_document(input=input_file_or_string, language='zh', normalization=None)
        self.candidate_selection(pos=pos)
        self.candidate_weighting(window=10, pos=pos)
        keyphrases = self.get_n_best(n=n_best)
        return keyphrases
