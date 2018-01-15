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
    the class for retaining the card_score_pair: [vocab word, field word, similarity]
    :param word:
    :param card_score_pair: [
    [Card_0, score with 'word'],
    [Card_1, score with 'word'],
    ...
    ]
    """

    def __init__(self, word, card_score_pair, team):

        self.word = word
        self.card_score_pair = card_score_pair
        self.team = team

        if team not in ["RED", "BLUE"]:
            raise ValueError("team name must be either RED or BLUE.")

        if team == "RED":
            self.enemy = "BLUE"
        else:
            self.enemy = "RED"

        self.total_score = self._calculate_score_with_threshold(card_score_pair)

    def _calculate_score_with_threshold(self, card_score_pair):
        """
        internal function.

        total_score = sigma(pos_score)
        pos_score is calculated from the cards whose similarity is greater than the one with negative cards.
        :param card_score_pair:
        :return: total_score
        """
        total_score = 0
        pos_score, neg_score = 0, 0
        pos_num, neg_num = 0, 0
        negative_list = [self.enemy, "NORMAL", "ASSASSIN"]

        sorted_card_score_pair = sorted(card_score_pair, key=lambda x: x[1], reverse=True)
        max_negative_score = -1
        for card, score in sorted_card_score_pair:
            # if card.taken_by is not None:
            if card.taken_by == "None":
                if card.color in negative_list:
                    if max_negative_score < score:
                        max_negative_score = score

        for card, score in sorted_card_score_pair:
            if card.taken_by == "None":
                if card.color in negative_list:
                    break
                else:
                    if score > max_negative_score:
                        total_score += score

        return total_score

    def _calculate_score(self, card_score_pair):
        """
        internal function to calculate the total score.

        total_score = sigma(pos_score)/len(positive_cards) - sigma(neg_score)/len(negative_cards)
        pos_score is the similarity for the field card which belongs to same team as
        the spymaster.
        neg_score is the similarity for the others.

        :param card_score_pair:
        :return: total_score
        """
        # need to consider variance...?
        total_score = 0
        pos_score, neg_score = 0, 0
        pos_num, neg_num = 0, 0

        for card, score in card_score_pair:
            if card.taken_by is not None:
            # if card.taken_by == "None":
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
                elif card.color == self.enemy:
                    neg_score += score
                    neg_num += 1

        total_score = pos_score/float(pos_num) - neg_score/float(neg_num)
        return total_score

    @staticmethod
    def print_word(Wr_class):
        """
        printing function for this class.

        :param Wr_class
        (must be the instance of this class)
        :return: None

        """
        print_text = "word: {}, total_score: {} \n".format(Wr_class.word, Wr_class.total_score)
        for card, score in Wr_class.card_score_pair:
            if card.taken_by == "None":
                card_score_text = "\t card.name:{} (team:{}), similarity: {} \n".format(card.name, card.color, score)
                print_text += card_score_text
            # print(print_text)
        return print_text

class Vocab(object):
    """
    my own vocabulary class.
    same structure is also implemented in gensim.models.vocab.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Spymaster(object):
    """
    Spymaster class.
    :param w2v_path: the path for pretrained embeddings
    :param field: the instance from Field class (defined in field.py)
    :param logger:
    :param team: team color (must be 'RED' or 'BLUE')
    :param word_table_path: path to save pkl file
    :param word_rank_list_path: path to save pkl file
    :param restrict_words_path: path to save pkl file
    :param test: bool value
    """

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
        self.fill_table()
        # print("table set.")

    def load_model(self, w2v_path):
        """
        load pretrained embeddings.
        :param w2v_path:
        :return:
        """

        self.logger.info("spymaster model loading...")
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.logger.info("spymaster model loaded.")
        return model

    def load_vocab(self):
        """
        load the vocabulary.
        by default, it loads all the vocabulary used for pretrained w2v (GoogleNews.bin.gz)
        the number should be 3000000.

        if you want to restrict the vocabulary, you can specify the path for the vocabulary
        as the self.restrict_word_path. (Recommended)

        :return: vocab (dict)
        """

        if len(self.restrict_words_path) == 0:
            self.logger.info("load vocabulary from gensim.models.KeyedVectors.")
            vocab = self.model.vocab
        else:
            vocab = dict()
            self.logger.info("load restrict words from csv: " + self.restrict_words_path)
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
        """
        fill the value for self.word_table.
        compute the similarity for each word from vocabulary and each word from field card.
        so the computation time is |vocabulary| * |field_size|(25).
        :return: None
        """
        self.logger.info("fill_table start.")
        word_table_path = self.word_table_path

        if os.path.exists(word_table_path):
            self.logger.info(word_table_path + " exists.")
            with open(word_table_path, 'rb') as r:
                self.word_table = pickle.load(r)

        else:
            self.logger.info("creating word_table...")
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
        self.logger.info("fill_table end.")

    def restrict_word(self):
        pass

    def give_clue_with_threshold(self, turn, top_n=100):
        """
        calculate the clue-likelihood for each word in vocabulary.
        with different scoring method.

        :param turn:
        :param top_n:
        :return:
        """
        word_rank_list_path = self.word_rank_list_path + str(turn)
        word_rank_list = []
        if os.path.exists(word_rank_list_path):
            self.logger.info(word_rank_list_path + " exists.")
            with open(word_rank_list_path, 'rb') as r:
                word_rank_list = pickle.load(r)
        else:
            self.logger.info("creating word_rank_list...")

            for word in self.vocab:
                card_score_pair = []
                word_ix = self.vocab[word].index
                for card in self.field:
                    score = self.word_table[word_ix][card.id]
                    card_score_pair.append([card, score])
                card_score_pair = sorted(card_score_pair, key=lambda x: x[1], reverse=True)

                a_wordrank = Wordrank(word, card_score_pair, team=self.team)
                word_rank_list.append(a_wordrank)

            word_rank_list = sorted(word_rank_list, key=lambda x: x.total_score, reverse=True)
            if not os.path.exists(word_rank_list_path):
                with open(word_rank_list_path, 'wb') as w:
                    pickle.dump(word_rank_list, w)
            else:
                self.logger.info(word_rank_list_path + " exists.")

            word_rank_list = word_rank_list[:top_n]

            for Wr_class in word_rank_list:
                self.logger.info(Wordrank.print_word(Wr_class))

            clue = word_rank_list[0].word
            possible_answers = word_rank_list[0].card_score_pair

            self.logger.info("clue: " + clue)

            num_count = 0
            count_ix = 0
            count_continue = True
            while count_continue:
                cur_card = word_rank_list[0].card_score_pair[count_ix][0]
                if cur_card.color in [self.team, "DOUBLE"]:
                    num_count += 1
                    count_ix += 1
                else:
                    count_continue = False

            self.logger.info("num: " + str(num_count))
            return clue, num_count, possible_answers

    def give_clue(self, turn, top_n=100):
        """
        calculate the clue-likelihood for each word in vocabulary
        :param turn: "RED" or "BLUE"
        :param top_n: the number you want to keep the word_rank_list for each combination
        :return:
        """

        # for subsets in 2^8:
        # sim(clue, perm) - sim(clue, others) for clue in models.word

        word_rank_list_path = self.word_rank_list_path + str(turn)

        pos_ix = [card.id for card in self.field if card.color == self.team and card.taken_by == "None"]
        neg_ix = [card.id for card in self.field if card.color != self.team and card.taken_by == "None"]
        neg_cards = [self.field[i] for i in neg_ix]

        # make all combination
        combinations = \
            [list(comb) for n_comb in range(1, len(pos_ix)+1)\
             for comb in itertools.combinations(pos_ix, n_comb)]

        word_rank_list = []

        # save model
        if os.path.exists(word_rank_list_path):
            self.logger.info(word_rank_list_path + " exists.")
            with open(word_rank_list_path, 'rb') as r:
                word_rank_list = pickle.load(r)
        else:
            self.logger.info("creating word_rank_list...")

            # brute force
            if self.test:
                # try small combination set
                self.logger.info("trying small combination set...")
                combinations = combinations[50:55]

            for comb in combinations:
                comb_cards = [self.field[i] for i in comb]

                sub_word_rank_list = []

                for word in self.vocab:
                    # too dirty
                    card_score_pair = []
                    word_ix = self.vocab[word].index
                    for card in comb_cards + neg_cards:
                        score = self.word_table[word_ix][card.id]
                        card_score_pair.append([card, score])

                    card_score_pair = sorted(card_score_pair, key=lambda x: x[1], reverse=True)

                    a_wordrank = Wordrank(word, card_score_pair, team=self.team)
                    sub_word_rank_list.append(a_wordrank)

                sub_word_rank_list = sorted(sub_word_rank_list, key=lambda x: x.total_score, reverse=True)

                # discard less than top_n
                top_n_sub_word_rank_list = sub_word_rank_list[:top_n]
                word_rank_list.extend(top_n_sub_word_rank_list)
                # self.logger.info("combination: ", comb, " ended.")

        # sort again
        word_rank_list = sorted(word_rank_list, key=lambda x: x.total_score, reverse=True)

        # save model
        if not os.path.exists(word_rank_list_path):
            with open(word_rank_list_path, 'wb') as w:
                pickle.dump(word_rank_list, w)
        else:
            self.logger.info(word_rank_list_path + " exists.")

        # limit the top
        word_rank_list = word_rank_list[:top_n]

        for Wr_class in word_rank_list:
            self.logger.info(Wordrank.print_word(Wr_class))

        clue = word_rank_list[0].word
        self.logger.info("clue: " + clue)

        num_count = 0
        count_ix = 0
        count_continue = True
        while count_continue:
            cur_card = word_rank_list[0].card_score_pair[count_ix][0]
            if cur_card.color == self.team:
                num_count += 1
                count_ix += 1
            else:
                count_continue = False

        self.logger.info("num: " + str(num_count))
        return clue, num_count


