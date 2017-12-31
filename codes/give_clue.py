import gensim
import itertools
import numpy as np
import pickle
import os
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

    # def __init__(self, word, field_card, score):
    def __init__(self, word, card_score_pair, team):
        '''

        :param word:
        :param card_score_pair: [
        [Card_0, score with 'word'],
        [Card_1, score with 'word'],
        ...
        ]
        '''
        self.word = word
        self.card_score_pair = card_score_pair
        self.team = team
        if team not in ["RED", "BLUE"]:
            raise ValueError("team name must be either RED or BLUE.")
        self.total_score = self.calculate_score(card_score_pair)

        # self.field_card = field_card
        # self.score = score

    def calculate_score(self, card_score_pair):
        # need to consider variance...?
        total_score = 0
        pos_score, neg_score = 0, 0
        pos_num, neg_num = 0, 0

        for card, score in card_score_pair:
            if card.color == self.team:
                pos_score += score
                pos_num += 1
            else:
                neg_score += score
                neg_num += 1
        total_score = pos_score/float(pos_num) - neg_score/float(neg_num)
        return total_score

    @staticmethod
    def print_word(Wr_class):
        print_text = "word: {}, total_score: {} \n".format(Wr_class.word, Wr_class.total_score)
        for card, score in Wr_class.card_score_pair:
            card_score_text = "\t card.name:{} (team:{}), similarity: {} \n".format(card.name, card.color, score)
            print_text += card_score_text
        print(print_text)
        # pass
        # print("word: {0}, score: {2}, pos_list:{1}".format(Wr_class.word, Wr_class.pos_ix, Wr_class.score))

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
        word_table_path = '../models/word_table.pkl'

        if os.path.exists(word_table_path):
            print(word_table_path, " exists.")
            with open(word_table_path, 'rb') as r:
                self.word_table = pickle.load(r)

        else:
            print("creating word_table...")
            for word in self.vocab:
                for card in self.field:
                    w_ix = self.vocab[word].index
                    c_ix = card.id
                    if word != card.name:
                        self.word_table[w_ix][c_ix] = \
                        self.model.similarity(word, card.name)
                    else:
                        self.word_table[w_ix][c_ix] = 0
            with open('../models/word_table.pkl', 'wb') as w:
                pickle.dump(self.word_table, w)
        print("fill_table end.")

    def give_clue(self, top_n=10):

        # for subsets in 2^8:
        # max(cossim(w, word in (subsets + negative))

        # suppose [c_0, c_1, ..., c_24]
        # positive_list [2, 4, 5, ..., 23]
        # others
        # give all permutation
        # for perm in permutation
        # sim(clue, perm) - sim(clue, others) for clue in models.word

        pos_ix = [card.id for card in self.field if card.color == self.team]
        neg_ix = [card.id for card in self.field if card.color != self.team]
        neg_cards = [self.field[i] for i in neg_ix]

        print("pos/neg set.")
        self.logger.info("pos/neg set.")

        # make all combination
        combinations = \
            [list(comb) for n_comb in range(1, len(pos_ix)+1)\
             for comb in itertools.combinations(pos_ix, n_comb)]
        print("combinations set.")
        self.logger.info("combinations set.")

        word_rank_list = []
        word_rank_list_path = "../models/wrl_100_ave.pkl"

        if os.path.exists(word_rank_list_path):
            with open(word_rank_list_path, 'rb') as r:
                word_rank_list = pickle.load(r)
        else:
            # brute force
            for comb in combinations:
                comb_cards = [self.field[i] for i in comb]

                print("combination: ", comb)

                sub_word_rank_list = []

                for word in self.vocab:
                    # too dirty
                    # modified
                    card_score_pair = []
                    word_ix = self.vocab[word].index
                    for card in comb_cards + neg_cards:
                        score = self.word_table[word_ix][card.id]
                        card_score_pair.append([card, score])

                    # score = self.word_table[word_ix][comb].sum()/len(comb) - \
                    #         self.word_table[word_ix][np.array(neg_ix)].sum()/len(neg_ix)

                    # a_wordrank = Wordrank(word, comb, score)
                    a_wordrank = Wordrank(word, card_score_pair, team='RED')
                    sub_word_rank_list.append(a_wordrank)

                sub_word_rank_list = sorted(sub_word_rank_list, key=lambda x: x.total_score, reverse=True)

                # discard less than top_n
                top_n_sub_word_rank_list = sub_word_rank_list[:top_n]
                word_rank_list.extend(top_n_sub_word_rank_list)
                print("combination: ", comb, " ended.")

        # sort again
        word_rank_list = sorted(word_rank_list, key=lambda x: x.total_score, reverse=True)
        word_rank_list_path = '../models/wrl_top100.pkl'
        if not os.path.exists(word_rank_list_path):
            with open(word_rank_list_path, 'wb') as w:
                pickle.dump(word_rank_list, w)
        else:
            print(word_rank_list_path, " exists.")

        for Wr_class in word_rank_list:
            Wordrank.print_word(Wr_class)

        print("clue:", word_rank_list[0].word)
        # max(cossim(w, word in (subsets + negative))

