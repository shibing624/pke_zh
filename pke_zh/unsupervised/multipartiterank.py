# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

Multipartite graph keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:

* Florian Boudin.
  Unsupervised Keyphrase Extraction with Multipartite Graphs.
  *In proceedings of NAACL*, pages 667-672, 2018.

"""
import math
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from pke_zh.unsupervised.topicrank import TopicRank


class MultipartiteRank(TopicRank):
    """Multipartite graph keyphrase extraction model."""

    def __init__(self, stopwords_path=None):
        """Redefining initializer for MultipartiteRank."""
        super(MultipartiteRank, self).__init__(stopwords_path)

        self.topic_identifiers = {}
        """ A container for linking candidates to topic identifiers. """

        self.graph = nx.DiGraph()
        """ Redefine the graph as a directed graph. """

    def clear_cache(self):
        self.graph.clear()
        self.topics.clear()
        self.topic_identifiers.clear()

    def topic_clustering(self, threshold=0.74, method='average'):
        """Clustering candidates into topics.

        :param threshold: float, the minimum similarity for clustering,
            defaults to 0.74, i.e. more than 1/4 of stem overlap
            similarity.
        :param method: str, the linkage method, defaults to average.
        """
        # handle document with only one candidate
        if len(self.candidates) == 1:
            candidate = list(self.candidates)[0]
            self.topics.append([candidate])
            self.topic_identifiers[candidate] = 0
            return

        # vectorize the candidates
        candidates, distance_matrix = self.vectorize_candidates()

        # compute the distance matrix
        jaccard_matrix = pdist(distance_matrix, 'jaccard')
        jaccard_matrix = np.nan_to_num(jaccard_matrix)

        # compute the clusters
        cluster_matrix = linkage(jaccard_matrix, method=method)

        # form flat clusters
        clusters = fcluster(cluster_matrix, t=threshold, criterion='distance')

        # for each cluster id
        for cluster_id in range(1, max(clusters) + 1):
            self.topics.append([candidates[j] for j in range(len(clusters))
                                if clusters[j] == cluster_id])

        # assign cluster identifiers to candidates
        for i, cluster_id in enumerate(clusters):
            self.topic_identifiers[candidates[i]] = cluster_id - 1

    def calculate_bi_node_weights(self, node_i, node_j, weights):
        """

        :param node_i: str
        :param node_j: str
        :param weights: list
        :return:
        """
        for p_i in self.candidates[node_i].offsets:
            for p_j in self.candidates[node_j].offsets:
                # compute gap
                gap = abs(p_i - p_j)

                # alter gap according to candidate length
                target_node = node_i if p_i < p_j else node_j
                gap -= len(self.candidates[target_node].lexical_form) - 1

                weights.append(1.0 / gap)

    def build_topic_graph(self):
        """ Build the Multipartite graph. """

        # adding the nodes to the graph
        self.graph.add_nodes_from(self.candidates.keys())

        # pre-compute edge weights
        for node_i, node_j in combinations(self.candidates.keys(), 2):

            # discard intra-topic edges
            if self.topic_identifiers[node_i] == self.topic_identifiers[node_j]:
                continue

            weights = []
            self.calculate_bi_node_weights(node_i, node_j, weights)

            # add weighted edges
            if weights:
                # node_i -> node_j
                self.graph.add_edge(node_i, node_j, weight=sum(weights))
                # node_j -> node_i
                self.graph.add_edge(node_j, node_i, weight=sum(weights))

    def is_valid_edge(self, variant, first, end):
        return variant != first and self.graph.has_edge(variant, end)

    def find_connected_nodes(self, first, variants, weighted_edges):
        """find the nodes to which it connects

        :param first:
        :param variants:
        :param weighted_edges: dict
        :return:
        """
        for start, end in self.graph.edges(first):
            boosters = [self.graph[v][end]['weight'] for v in variants
                        if self.is_valid_edge(v, first, end)]

            if boosters:
                weighted_edges[(start, end)] = np.sum(boosters)

    def weight_adjustment(self, alpha=1.1):
        """ Adjust edge weights for boosting some candidates.

        :param alpha: float, hyper-parameter that controls the strength of the
            weight adjustment, defaults to 1.1.
        """
        weighted_edges = {}

        # Topical boosting
        for variants in self.topics:
            # skip one candidate topics
            if len(variants) == 1:
                continue

            # get the offsets
            offsets = [self.candidates[v].offsets[0] for v in variants]

            # get the first occurring variant
            first = variants[offsets.index(min(offsets))]

            # find the nodes to which it connects
            self.find_connected_nodes(first, variants, weighted_edges)

        # update edge weights
        for nodes, boosters in weighted_edges.items():
            node_i, node_j = nodes
            position_i = 1.0 / (1 + self.candidates[node_i].offsets[0])
            position_i = math.exp(position_i)
            self.graph[node_j][node_i]['weight'] += (boosters * alpha * position_i)

    def candidate_weighting(self,
                            threshold=0.74,
                            method='average',
                            alpha=1.1):
        """Candidate weight calculation using random walk.

        :param threshold: float, the minimum similarity for clustering, defaults to 0.25.
        :param method: str, the linkage method, defaults to average.
        :param alpha: float, hyper-parameter that controls the strength of the
            weight adjustment, defaults to 1.1.
        """
        if not self.candidates:
            return

        # cluster the candidates
        self.topic_clustering(threshold=threshold, method=method)

        # build the topic graph
        self.build_topic_graph()

        if alpha > 0.0:
            self.weight_adjustment(alpha)

        # compute the word scores using random walk
        self.weights = nx.pagerank_scipy(self.graph)

    def extract(self, input_file_or_string, n_best=10, pos=None, threshold=0.74):
        # 1. load document
        self.load_document(input=input_file_or_string, language='zh')

        # 2. clear previous graph and topics if exists
        self.clear_cache()

        # 3. select the longest sequences of nouns and adjectives, that do
        # not contain punctuation marks or stopwords as candidates.
        if pos is None:
            pos = {'n', 'a'}
        self.candidate_selection(pos=pos)

        # 4. build the Multipartite graph and rank candidates using random walk,
        # alpha controls the weight adjustment mechanism, see TopicRank for
        # threshold/method parameters.
        self.candidate_weighting(alpha=1.1,
                                 threshold=threshold,
                                 method='average')

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = self.get_n_best(n=n_best)
        return keyphrases
