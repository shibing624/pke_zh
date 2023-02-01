# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

refer: https://maartengr.github.io/KeyBERT/guides/quickstart.html
keybert关键词抽取，核心思想类似embedrank，
    只是向量提取的度量方法使用keyBERT
"""

from loguru import logger
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from text2vec import SentenceModel
import numpy as np
from typing import List

from pke_zh.base import BaseKeywordExtractModel


def max_sum_ranking(doc_embedding: np.ndarray,
                    can_embeddings: np.ndarray,
                    can_names: List[str],
                    top_n: int,
                    nr_candidates: int):
    """ Calculate Max Sum Distance for extraction of keywords
        We take the 2 x top_n most similar words/phrases to the document.
        Then, we take all top_n combinations from the 2 x top_n words and
        extract the combination that are the least similar to each other
        by cosine similarity.
        NOTE:
            This is O(n^2) and therefore not advised if you use a large top_n
        Arguments:
            doc_embedding: The document embeddings
            can_embeddings: The embeddings of the selected candidate keywords/phrases
            can_names: The selected candidate keywords/keyphrases
            top_n: 取top_n 个关键词
            nr_candidates: The number of candidates to consider, generaly set top_n *2
        Returns:
             List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
        """
    # calculate distances and extract words
    # print(doc_embedding)
    distances = cosine_similarity(doc_embedding, can_embeddings)
    distance_words = cosine_similarity(can_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    can_idx = list(distances.argsort()[0][-nr_candidates:])

    can_name_filter = [can_names[i] for i in can_idx]
    cand_distance = distance_words[np.ix_(can_idx, can_idx)]

    # Calculate the 候选词里的topn of words的组合， that are the least similar to each other
    min_sim = 100000
    final_candidate = None
    # print(can_idx)
    for combination in itertools.combinations(range(len(can_idx)), top_n):
        sim = sum([cand_distance[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            final_candidate = combination
            min_sim = sim
    # return candi_name and score
    result = []
    if not final_candidate:
        final_candidate = can_idx
    for val in final_candidate:
        result.append((can_name_filter[val], distances[0][can_idx[val]]))
    return result


def mmr_ranking(doc_embedding: np.ndarray,
                can_embeddings: np.ndarray,
                can_names: List[str],
                top_n: int,
                alpha: float = 0.5):
    """ Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.
    Arguments:
        doc_embedding: The document embeddings
        can_embeddings: The embeddings of the selected candidate keywords/phrases
        can_names: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        alpha: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    # calculate distances and extract words
    doc_can_distances = cosine_similarity(can_embeddings, doc_embedding)
    distance_words = cosine_similarity(can_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(doc_can_distances)]
    candidates_idx = [i for i in range(len(can_names)) if i != keywords_idx[0]]

    for r in range(min(top_n, len(can_embeddings) - 1)):
        # extract similarities
        candidate_similarities = doc_can_distances[candidates_idx, :]
        target_similarities = np.max(distance_words[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = alpha * candidate_similarities - (1 - alpha) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # return candia_name and score
    result = []
    for val in keywords_idx:
        result.append((can_names[val], doc_can_distances[val][0]))
    return result


def mmr_norm_ranking(doc_embedding: np.ndarray,
                     can_embeddings: np.ndarray,
                     can_names: List[str],
                     top_n: int,
                     alpha: float = 0.5):
    """Rank candidates according to a query

    :param document: np.array, dense representation of document (query)
    :param candidates: np.array, dense representation of candidates
    :param l: float, ratio between distance to query or distance between
        chosen candidates
    Returns: a list of candidates rank
    """

    def norm(sim, **kwargs):
        sim -= sim.min(**kwargs)
        sim /= (sim.max(**kwargs) + 1e-10)
        sim = 0.5 + (sim - sim.mean(**kwargs)) / (sim.std(**kwargs) + 1e-10)
        return sim

    def norm2(sim, **kwargs):
        min_ = sim.min(**kwargs)
        max_ = (sim.max(**kwargs) + 1e-10)
        sim = (sim - min_) / max_
        sim = 0.5 + (sim - sim.mean(**kwargs)) / (sim.std(**kwargs) + 1e-10)
        return sim

    sim_doc = cosine_similarity(doc_embedding, can_embeddings)
    sim_doc[np.isnan(sim_doc)] = 0.
    sim_doc = norm(sim_doc)
    sim_doc[np.isnan(sim_doc)] = 0.

    sim_can = cosine_similarity(can_embeddings)
    sim_can[np.isnan(sim_can)] = 0.
    sim_can = norm(sim_can, axis=1)
    sim_can[np.isnan(sim_can)] = 0.

    sel = np.zeros(len(can_embeddings), dtype=bool)
    ranks = [None] * len(can_embeddings)
    # Compute first candidate, the second part of the calculation is 0
    # as there are no other chosen candidates to maximise distance to
    chosen_candidate = (sim_doc * alpha).argmax()
    sel[chosen_candidate] = True
    ranks[chosen_candidate] = 0

    for r in range(1, len(can_embeddings)):
        # Remove already chosen candidates
        sim_can[sel] = np.nan

        # Compute MMR score
        scores = alpha * sim_doc - (1 - alpha) * sim_can[:, sel].max(axis=1)
        chosen_candidate = np.nanargmax(scores)

        # Update output and mask with chosen candidate
        sel[chosen_candidate] = True
        ranks[chosen_candidate] = r

    result = []
    for can_id, val in enumerate(ranks):
        if not val is None:
            result.append((can_names[can_id], (len(ranks) - 1 - val) / (len(ranks) - 1)))

    return result


class KeyBert(BaseKeywordExtractModel):
    def __init__(self, model='shibing624/text2vec-base-chinese'):
        """
            原文支持若干种embedding 方法：SentenceModel、SentenceTransformers、Flair、Spacy、gensim
            中文默认支持text2vec.SentenceModel model="shibing624/text2vec-base-chinese"模型，
            英文可设置model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"模型，
            其他语言参考：sentence-transformers models:
                  * https://www.sbert.net/docs/pretrained_models.html
            """
        # param: model sentenceTransformers
        super(KeyBert, self).__init__()
        if isinstance(model, str):
            try:
                self.model = SentenceModel(model)
            except Exception as e:
                logger.error('wrong url for sentence model, change to default!')
                self.model = SentenceModel('shibing624/text2vec-base-chinese')
        elif isinstance(model, SentenceModel):
            self.model = model
        else:
            raise ValueError('model must be str or text2vec.SentenceModel')
        self.max_length = self.model.max_seq_length

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.
        :param pos: set of valid POS tags, defaults to ('NOUN', 'ADJ').
        """
        if pos is not None:
            self.valid_pos = pos

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=self.valid_pos)
        self.candidate_filtering()

    def _flatten_doc_words(self, lower):
        """flatten sentence words whose postags are valid"""
        doc = ' '.join(w.lower() if lower else w for s in self.sentences
                       for i, w in enumerate(s.words)
                       if s.pos[i] in self.valid_pos)
        return doc

    def _calculate_candidate_weights(self, rank, cand_name):
        for candidate_id, r in enumerate(rank):
            if len(rank) > 1:
                # Inverting ranks so the first ranked candidate has the biggest score
                score = (len(rank) - 1 - r) / (len(rank) - 1)
            else:
                score = r
            self.weights[cand_name[candidate_id]] = score

    def _doc_to_sent_list(self):
        """
        为了充分获取长文本语义，针对单句做的合并
        :return: sentences list
        """
        sentence = []
        cur_len = 0
        cur_sent = []
        for i, sent in enumerate(self.sentences):
            cur_text = ''.join(sent.words)
            cur_len += len(cur_text)
            if cur_len >= self.max_length and cur_sent:
                sentence.append(''.join(cur_sent))
                cur_sent = [cur_text]
                cur_len = len(cur_text)
            else:
                cur_sent.append(cur_text)
        if cur_len:
            sentence.append("".join(cur_sent))
        return sentence

    def _weights_update(self, canlist):
        for canname, score in canlist:
            self.weights[canname] = score

    def candidate_weighting(self, use_maxsum=True, use_mmr=False, top_n=10, alpha=0.5, nr_candidates=20):
        """Candidate weighting function using distance to document.

        :param l: float, Lambda parameter for EmbedRank++ Maximal Marginal
        Relevance (MMR) computation. Use 1 to compute EmbedRank and 0 to not
        use the document, but only the most diverse set of candidates
        (defaults to 1).
        """
        # get doc's sentences
        doc_sents = self._doc_to_sent_list()
        doc_embed = self.model.encode(doc_sents)
        doc_embed = np.average(doc_embed, axis=0)  # 取平均
        doc_embed = np.expand_dims(doc_embed, axis=0)  # 增加一个维度

        cand_name = list(self.candidates.keys())  # 记得是带空格的
        cand_embed = self.model.encode(cand_name)

        if use_mmr:
            can_list = mmr_ranking(doc_embed, cand_embed, cand_name, top_n, alpha)
            self._weights_update(can_list)
        elif use_maxsum:
            can_list = max_sum_ranking(doc_embed, cand_embed, cand_name, top_n, nr_candidates)
            self._weights_update(can_list)
        else:
            distances = cosine_similarity(doc_embed, cand_embed)
            for i, score in enumerate(distances[0]):
                self.weights[cand_name[i]] = score

    def extract(self, input_file_or_string, n_best=10, use_maxsum=True, use_mmr=False, alpha=0.5, nr_candidates=20):
        """
        提取text的关键词
        :param input_file_or_string: 输入doc
        :param n_best: top n_best个关键词
        :param use_maxsum: 是否使用maxsum similarity for the selection of keywords
        :param use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the selection of keywords/keyphrases
        :param alpha: mmr算法的超参数，if use_mmr is set True, default:0.5
        :param nr_candidates: The number of candidates to consider if use_maxsum is set to True
        :return: keywords list
        """
        # 1. load the content of the document.
        self.load_document(input=input_file_or_string, language='zh', normalization=None)
        # 2. select sequences of nouns and adjectives as candidates.
        self.candidate_selection()
        # 3. weight the candidates using EmbedRank method
        self.candidate_weighting(use_maxsum, use_mmr, n_best, alpha, nr_candidates)
        # 4. get the 10-highest scored candidates as keyphrases
        keyphrases = self.get_n_best(n=n_best, redundancy_removal=True)
        return keyphrases
