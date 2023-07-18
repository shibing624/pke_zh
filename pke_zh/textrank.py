# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

TextRank keyphrase extraction model.

Implementation of the TextRank model for keyword extraction described in:

* Rada Mihalcea and Paul Tarau.
  TextRank: Bringing Order into Texts
  *In Proceedings of EMNLP*, 2004.

Extract sentence references from:
    https://github.com/skykiseki/textrank4ch/blob/ed6a295139ebd38ba494182f8c2634bedfafb14e/textrank4ch/utils.py
"""

import re
import math
import networkx as nx
import numpy as np
from pke_zh.utils.text_utils import edit_distance
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
        :param pos: set of valid POS tags, defaults to ('NOUN', 'ADJ').
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
            to_keep = max(1, to_keep)

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

    def extract_sentences(self, doc_string, n_best=10):
        """Extract key sentences from a text.
        :param doc_string: str, the input text.
        :param n_best: int, the number of keysentences to return.
        """
        keysentences = []
        if not doc_string:
            return keysentences
        list_res = []
        content = doc_string.replace(" ", "")

        # split to sentences
        pattern = '[{0}*]'.format('|'.join(self.sentence_delimiters))
        sentences = [s for s in re.split(pattern, content) if len(s) > 0]
        len_sentences = len(sentences)
        # 初始化句子之间的无向权图, 整体为N*N的矩阵
        graph = np.zeros((len_sentences, len_sentences))

        # 计算权重, 权重由切词的相似度进行计算, 由于是无向的, a(ij) = a(ji)
        for i in range(len_sentences):
            for j in range(len_sentences):
                sim_value = 1.0 - edit_distance(sentences[i], sentences[j])
                graph[i, j] = sim_value
                graph[j, i] = sim_value

        # 构造无向权图
        nx_graph = nx.from_numpy_matrix(graph)

        # 默认的PR收敛时的参数
        pr_alpha = 1
        pr_max_iter = 200
        pr_tol = 1e-6
        pagerank_config = {'alpha': pr_alpha,
                           'max_iter': pr_max_iter,
                           'tol': pr_tol}
        pr_values = None
        # 计算PR值, 注意, 初始参数在计算PR值时可能不收敛, 这个时候可以
        flag = True
        while flag:
            try:
                # 开始计算PR值, 可能存在不收敛的情况
                pr_values = nx.pagerank(nx_graph, **pagerank_config)
                # 成功收敛则停止循环
                flag = False
            except Exception:
                # 如果PR不收敛, 以提升迭代前后轮次之间的差值为策略，也提升迭代轮次
                pr_tol *= 10
                pr_max_iter += 100

                pagerank_config = {'alpha': pr_alpha,
                                   'max_iter': pr_max_iter,
                                   'tol': pr_tol}

        # pr_values: 一个dict, {index:pr, index:pr}
        for idx, val in sorted(pr_values.items(), key=lambda x: x[1], reverse=True):
            list_res.append({'sentence': sentences[idx],
                             'weight': val,
                             'index': idx})
        keysentences = [(item['sentence'], item['weight']) for item in list_res][:n_best]
        return keysentences

    def extract(self, input_file_or_string, n_best=10, pos=None, top_percent=0.33):
        """Extract keyphrases from a text or a file.
        :param input_file_or_string: str, the input text or file path.
        :param n_best: int, the number of keyphrases to return.
        :param pos: the set of valid pos for words to be considered as nodes
            in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        :param top_percent: float, percentage of top vertices to keep for phrase generation.
        """
        keyphrases = []
        if not input_file_or_string:
            return keyphrases
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
