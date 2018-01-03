import gensim
import itertools
import numpy as np
import pickle
import os
import csv

from functools import reduce
import random
import time
import sys

class Wordrank(object):

    """
    params
    self.word: target word
    self.pos: list( (positive_card_name, cossim_score) )
    self.neg: list( (negative_card_name, cossim_score) )
    self.score: score of target word
    """

    def __init__(self, word, card_score_pair, team):
        """
        :param word:
        :param card_score_pair: [
        [Card_0, score with 'word'],
        [Card_1, score with 'word'],
        ...
        ]
        """

        self.word = word
        self.card_score_pair = card_score_pair
        self.team = team

        if team not in ["RED", "BLUE"]:
            raise ValueError("team name must be either RED or BLUE.")

        if team == "RED":
            self.enemy = "BLUE"
        else:
            self.enemy = "RED"

        self.total_score = self.calculate_score(card_score_pair)

    def calculate_score(self, card_score_pair):
        # need to consider variance...?
        total_score = 0
        pos_score, neg_score = 0, 0
        pos_num, neg_num = 0, 0

        for card, score in card_score_pair:
            if card.color == self.team:
                pos_score += score
                pos_num += 1

            elif card.color == "DOUBLE":
                neg_score += score
                neg_num += 1

            elif card.color == "NORMAL":
                neg_score += score
                neg_num += 1
            elif card.color == "ASSASSIN":
                neg_score += score
                neg_num += 1
            elif card.score == self.enemy:
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
        # print("word: {0}, score: {2}, pos_list:{1}".format(Wr_class.word, Wr_class.pos_ix, Wr_class.score))

class Vocab(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Spymaster(object):
    def __init__(self, w2v_path, field, logger, team,
                 word_table_path, word_rank_list_path, restrict_words_path,
                 test=False):
        self.test = test
        self.w2v_path = w2v_path
        self.word_table_path = word_table_path
        self.word_rank_list_path = word_rank_list_path
        self.restrict_words_path = restrict_words_path
        self.field = field
        self.logger = logger

        if team not in ["RED", "BLUE"]:
            raise ValueError("team string must be RED or BLUE.")
        self.team = team

        self.model = self.load_model(self.w2v_path)
        self.vocab = self.load_vocab()
        self.vocab_size = len(self.vocab)

        # initialize by 2
        # 25 * 3000000 due to memory limitation
        self.word_table = np.full([self.vocab_size, len(self.field)], 2, dtype=np.float32)
        # print("self.word_table.shape: ", self.word_table.shape)
        self.fill_table()
        print("table set.")

    def load_model(self, w2v_path):
        print("model loading...")
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        print("model loaded.")
        return model

    def load_vocab(self):
        if len(self.restrict_words_path) == 0:
            print("load vocabulary from gensim.models.KeyedVectors.")
            vocab = self.model.vocab
        else:
            vocab = dict()
            print("load restrict words from csv: ", self.restrict_words_path)
            with open(self.restrict_words_path, 'r') as c:
                reader = csv.reader(c)
                header = next(reader)
                cur_id = 0
                for row in reader:
                    word = row[1].lower()
                    if word not in vocab:
                        vocab[word] = Vocab(index=cur_id)
                        cur_id += 1
        return vocab

    def fill_table(self):
        print("fill_table start.")
        word_table_path = self.word_table_path

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

                    if word not in self.model.vocab:
                        self.word_table[w_ix][c_ix] = 0.0
                    else:
                        if word != card.name:
                            self.word_table[w_ix][c_ix] = self.model.similarity(word, card.name)
                        else:
                            self.word_table[w_ix][c_ix] = 0
            with open('../models/word_table.pkl', 'wb') as w:
                pickle.dump(self.word_table, w)
        print("fill_table end.")

    def restrict_word(self):
        pass

    def give_clue(self, top_n=100):

        # for subsets in 2^8:
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

        # save model
        if os.path.exists(self.word_rank_list_path):
            print(self.word_rank_list_path, "exists.")
            with open(self.word_rank_list_path, 'rb') as r:
                word_rank_list = pickle.load(r)
        else:
            print("creating word_rank_list...")

            # brute force
            if self.test:
                # try small combination set
                print("trying small combination set...")
                combinations = combinations[50:55]

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

                    a_wordrank = Wordrank(word, card_score_pair, team='RED')
                    sub_word_rank_list.append(a_wordrank)

                sub_word_rank_list = sorted(sub_word_rank_list, key=lambda x: x.total_score, reverse=True)

                # discard less than top_n
                top_n_sub_word_rank_list = sub_word_rank_list[:top_n]
                word_rank_list.extend(top_n_sub_word_rank_list)
                print("combination: ", comb, " ended.")

        # sort again
        word_rank_list = sorted(word_rank_list, key=lambda x: x.total_score, reverse=True)

        # save model
        if not os.path.exists(self.word_rank_list_path):
            with open(self.word_rank_list_path, 'wb') as w:
                pickle.dump(word_rank_list, w)
        else:
            print(self.word_rank_list_path, " exists.")

        for Wr_class in word_rank_list:
            Wordrank.print_word(Wr_class)

        print("clue: ", word_rank_list[0].word, "num: ", word_rank_list[0].card_score_pair)
        num_count = 0
        count_ix = 0
        count_continue = True
        while(count_continue):
            cur_card = word_rank_list[0].card_score_pair[count_ix][0]
            if cur_card.color == self.team:
                num_count += 1
                count_ix += 1
            else:
                count_continue = False
        print("num: ", num_count)

        # max(cossim(w, word in (subsets + negative))

