# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Florian Boudin
@description:

TopicRank keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:

* Adrien Bougouin, Florian Boudin and Béatrice Daille.
  TopicRank: Graph-Based Topic Ranking for Keyphrase Extraction.
  *In proceedings of IJCNLP*, pages 543-551, 2013.

"""

from itertools import combinations

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from pke_zh.base import BaseKeywordExtractModel


class TopicRank(BaseKeywordExtractModel):
    """TopicRank keywords extraction model."""

    def __init__(self, stopwords_path=None):
        """Redefining initializer for TopicRank."""

        super(TopicRank, self).__init__(stopwords_path)

        self.graph = nx.Graph()
        """ The topic graph. """

        self.topics = []
        """ The topic container. """

    def clear_cache(self):
        """clear previous graph and topics if exists"""
        self.graph.clear()
        self.topics.clear()

    def candidate_selection(self, pos=None, stoplist=None):
        """Selects longest sequences of nouns and adjectives as keyphrase candidates.

        :param pos: the set of valid POS tags, defaults to ('NOUN', 'ADJ').
        :param stoplist: the stoplist for filtering candidates, defaults to
            the nltk stoplist. Words that are punctuation marks from
            string.punctuation are not allowed.
        """
        # define default pos tags set
        if pos is None:
            pos = {'n', 'a'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)

        # initialize stoplist list if not provided
        if stoplist is None:
            stoplist = self.stoplist

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=stoplist)

    def vectorize_candidates(self):
        """Vectorize the keyphrase candidates.

        Returns:
            candidate_list (list): the list of candidates.
            candidate_matrix (matrix): vectorized representation of the candidates.
        """
        # build the vocabulary, i.e. setting the vector dimensions
        dim = set([])
        # for k, v in self.candidates.iteritems():
        # iterate Python 2/3 compatible
        for (k, v) in self.candidates.items():
            for w in v.lexical_form:
                dim.add(w)
        dim = list(dim)

        # vectorize the candidates Python 2/3 + sort for random issues
        candidate_list = list(self.candidates)
        candidate_list.sort()

        candidate_matrix = np.zeros((len(candidate_list), len(dim)))
        for i, k in enumerate(candidate_list):
            for w in self.candidates[k].lexical_form:
                candidate_matrix[i, dim.index(w)] += 1

        return candidate_list, candidate_matrix

    def topic_clustering(self, threshold=0.74, method='average'):
        """Clustering candidates into topics.

        :param threshold: float, the minimum similarity for clustering, defaults
            to 0.74, i.e. more than 1/4 of stem overlap similarity.
        :param method: str, the linkage method, defaults to average.
        """
        # handle document with only one candidate
        if len(self.candidates) == 1:
            self.topics.append([list(self.candidates)[0]])
            return

        # vectorize the candidates
        candidates, distance_matrix = self.vectorize_candidates()

        # compute the distance matrix
        jaccard_matrix = pdist(distance_matrix, 'jaccard')

        # compute the clusters
        cluster_matrix = linkage(jaccard_matrix, method=method)

        # form flat clusters
        clusters = fcluster(cluster_matrix, t=threshold, criterion='distance')

        # for each topic identifier
        for cluster_id in range(1, max(clusters) + 1):
            self.topics.append([candidates[j] for j in range(len(clusters))
                                if clusters[j] == cluster_id])

    @staticmethod
    def fetch_topic_index(topic_i, topic_j, offset_i, offset_j):
        if offset_i < offset_j:
            return topic_i
        else:
            return topic_j

    def calculate_topics_weight(self, ti, tj):
        """
        Calculate the weight of the topic.
        :param ti: topic_idx_i
        :param tj: topic_idx_j
        :return:
        """
        for c_i in self.topics[ti]:
            for c_j in self.topics[tj]:
                for p_i in self.candidates[c_i].offsets:
                    for p_j in self.candidates[c_j].offsets:
                        gap = abs(p_i - p_j)
                        target_c = self.fetch_topic_index(c_i, c_j, p_i, p_j)
                        gap -= len(self.candidates[target_c].lexical_form) - 1
                        self.graph[ti][tj]['weight'] += 1.0 / gap

    def build_topic_graph(self):
        """Build topic graph."""
        # adding the nodes to the graph
        self.graph.add_nodes_from(range(len(self.topics)))

        # loop through the topics to connect the nodes
        for i, j in combinations(range(len(self.topics)), 2):
            self.graph.add_edge(i, j, weight=0.0)
            self.calculate_topics_weight(i, j)

    def candidate_weighting(
            self,
            threshold=0.74,
            method='average',
            heuristic=None
    ):
        """Candidate ranking using random walk.

        :param threshold: float, the minimum similarity for clustering, defaults to 0.74.
        :param method: str, the linkage method, defaults to average.
        :param heuristic: str, the heuristic for selecting the best candidate for
            each topic, defaults to first occurring candidate. Other options
            are 'frequent' (most frequent candidate, position is used for
            ties).
        """
        if not self.candidates:
            return

        # cluster the candidates
        self.topic_clustering(threshold=threshold, method=method)

        # build the topic graph
        self.build_topic_graph()

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph, alpha=0.85, weight='weight')

        # loop through the topics
        for i, topic in enumerate(self.topics):

            # get the offsets of the topic candidates
            offsets = [self.candidates[t].offsets[0] for t in topic]

            # get first candidate from topic
            if heuristic == 'frequent':

                # get frequencies for each candidate within the topic
                freq = [len(self.candidates[t].surface_forms) for t in topic]

                # get the indexes of the most frequent candidates
                indexes = [j for j, f in enumerate(freq) if f == max(freq)]

                # offsets of the indexes
                indexes_offsets = [offsets[j] for j in indexes]
                # Choosing the first occuring most frequent candidate
                most_frequent = offsets.index(min(indexes_offsets))
                self.weights[topic[most_frequent]] = w[i]

            else:
                first = offsets.index(min(offsets))
                self.weights[topic[first]] = w[i]

    def extract(self, input_file_or_string, n_best=10, pos=None):
        if pos is None:
            pos = {'n', 'a'}
        self.load_document(input=input_file_or_string, language='zh')
        self.clear_cache()
        self.candidate_selection(pos=pos)
        self.candidate_weighting(threshold=0.74, method='average')
        # get the 10-highest scored candidates as keyphrases
        keyphrases = self.get_n_best(n=n_best)
        return keyphrases
