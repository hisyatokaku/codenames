import gensim
import itertools
import numpy as np
import pickle
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

    '''
    def __init__(self, word, pos_list, neg_list):
        self.word = word
        self.pos_list = pos_list
        self.neg_list = neg_list
        self.score = self.calculate_score()
    '''

    def __init__(self, word, pos_ix, score):
        self.word = word
        self.pos_ix = pos_ix
        self.score = score

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
        print("word: {0}, score: {1}, pos_list:{2}".format(Wr_class.word, Wr_class.pos, \
                                                                        Wr_class.score))

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
        self.vocab = self.model.vocab
        self.vocab_size = len(self.vocab)

        # initialize by 2
        # 25 * 3000000 due to memory limitation
        self.word_table = np.full([self.vocab_size, len(self.field)], 2, dtype=np.float32)
        self.fill_table()
        print("table set.")

    def load_model(self, w2v_dir):
        print("model loading...")
        if self.test:
            model = None
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=True)
        print("model loaded.")
        return model

    """
    def word_similarity(self, vocab_word, card):
        index_1 = self.vocab[vocab_word].index
        index_2 = card.id
        if self.word_table[index_1][index_2] == 2.0:
            self.word_table[index_1][index_2] = self.model.similarity(vocab_word, card.name)

        return float(self.word_table[index_1][index_2])
    """

    def fill_table(self):
        print("fill_table start.")

        for word in self.vocab:
            for card in self.field:
                w_ix = self.vocab[word].index
                c_ix = card.id
                if w_ix != c_ix:
                    self.word_table[w_ix][c_ix] = \
                    self.model.similarity(word, card.name)
                else:
                    self.word_table[w_ix][c_ix] = 0
        print("fill_table end.")

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

        print("pos/neg set.")
        self.logger.info("pos/neg set.")
        # make all combination
        combinations = \
            [list(comb) for n_comb in range(1, len(pos_ix)+1)\
             for comb in itertools.combinations(pos_ix, n_comb)]
        print("combinations set.")
        self.logger.info("combinations set.")

        word_rank_list = []
        # brute force
        for comb in combinations:
            print("combination: ", comb)

            # pos_card_list = [self.field[i] for i in comb]
            # neg_card_list = [self.field[i] for i in neg_ix]
            sub_word_rank_list = []

            for word in self.vocab:
                # too dirty
                # pos_similarities = [(card.name, self.word_similarity(word, card)) for card in pos_card_list]
                # neg_similarities = [(card.name, self.word_similarity(word, card)) for card in neg_card_list]

                # a_wordrank = Wordrank(word, pos_similarities, neg_similarities)
                # modified
                word_ix = self.vocab[word].index

                score = self.word_table[word_ix][comb].sum() - \
                        self.word_table[word_ix][np.array(neg_ix)].sum()

                a_wordrank = Wordrank(word, comb, score)
                sub_word_rank_list.append(a_wordrank)

            sub_word_rank_list = sorted(sub_word_rank_list, key=lambda x: x.score, reverse=True)

            # discard less than top_n
            top_n_sub_word_rank_list = sub_word_rank_list[:top_n]
            word_rank_list.extend(top_n_sub_word_rank_list)
            print("combination: ", comb, " ended.")

        with open('../models/wrl.pkl', 'wb') as w:
            pickle.dump(word_rank_list, w)

        # sort again
        word_rank_list = sorted(word_rank_list, key=lambda x: x.score, reverse=True)

        for Wr_class in word_rank_list:
            Wordrank.print_word(Wr_class)

        print("clue:", word_rank_list[0].word)

        # max(cossim(w, word in (subsets + negative))

