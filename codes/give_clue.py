import gensim
import itertools
from functools import reduce
import random
import time
import sys

class Wordrank(object):
    """
    member

    self.word: target word
    self.pos: list( (positive_card_name, cossim_score) )
    self.neg: list( (negative_card_name, cossim_score) )
    self.score: score of target word
    """

    def __init__(self, word, pos_list, neg_list):
        self.word = word
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.score = self.calculate_score()

    def calculate_score(self):
        # need to consider variance...?
        # score = reduce(lambda a, b: a[1]+b[1], self.pos_list)/len(self.pos_list)\
        #         - reduce(lambda a, b: a[1]+b[1], self.neg_list)/len(self.neg_list)
        score = 0
        for pos in self.pos_list:
            score += pos[1]/float(len(self.pos_list))
        for neg in self.neg_list:
            score -= neg[1]/float(len(self.neg_list))

        return score

    @staticmethod
    def print_word(Wr_class):
        print("word: {0}, score: {3}, pos_list:{1}, neg_list:{2}".format(Wr_class.word, Wr_class.pos, \
                                                                        Wr_class.neg, Wr_class.score))

class Spymaster(object):
    def __init__(self, w2v_dir, field, logger, team, test=False):
        self.test = test
        self.w2v_dir = w2v_dir
        self.field = field
        self.logger = logger
        if team not in ["RED", "BLUE"]:
            raise ValueError("team string must be RED or BLUE.")
        self.team = team
        self.model = self.load_model(self.w2v_dir)

    def load_model(self, w2v_dir):
        print("model loading...")
        if self.test:
            model = None
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=True)
        print("model loaded.")
        return model

    def give_clue(self, top_n=10):

        # for subsets in 2^8:
        # max(cossim(w, word in (subsets + negative))

        # suppose [c_1, c_2, ..., c_25]
        # positive_list [2, 4, 5, ..., 23]
        # others
        # give all permutation
        # for perm in permutation
        # sim(clue, perm) - sim(clue, others) for clue in models.word

        pos_ix = [card.id for card in self.field if card.color == self.team]
        neg_ix = [card.id for card in self.field if card.color != self.team]

        # make all combination
        combinations = \
            [list(comb) for n_comb in range(1, len(pos_ix)+1)\
             for comb in itertools.combinations(pos_ix, n_comb)]

        word_rank_list = []

        # brute force
        for comb in combinations:
            pos_card_list = [self.field[i] for i in comb]
            neg_card_list = [self.field[i] for i in neg_ix]
            vocab = self.model.vocab
            sub_word_rank_list = []
            for word in vocab:
                pos_similarities = [(card.name, self.model.similarity(word, card.name)) for card in pos_card_list]
                neg_similarities = [(card.name, self.model.similarity(word, card.name)) for card in neg_card_list]

                # pos_score = map(lambda x: x[1]/len(pos_similarities), pos_similarities)
                # neg_score = map(lambda x: x[1]/len(neg_similarities), neg_similarities)

                a_wordrank = Wordrank(word, pos_similarities, neg_similarities)
                sub_word_rank_list.append(a_wordrank)

            sub_word_rank_list = sorted(sub_word_rank_list, key=lambda x: x.score, reverse=True)

            # discard less than top_n
            top_n_sub_word_rank_list = sub_word_rank_list[:top_n]
            word_rank_list.append(top_n_sub_word_rank_list)

        # sort again
        word_rank_list = sorted(word_rank_list, key=lambda x: x.score, reverse=True)

        for Wr_class in word_rank_list:
            Wordrank.print_word(Wr_class)

        print("clue:", word_rank_list[0].word)

        # max(cossim(w, word in (subsets + negative))

