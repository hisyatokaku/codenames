import random
import gensim
import time
import sys
from field import Card, Field


class Guesser(object):
    def __init__(self, w2v_dir, field, logger, test=False):
        self.test = test
        self.w2v_dir = w2v_dir
        self.field = field
        self.logger = logger
        self.model = self.load_model(self.w2v_dir)

    def load_model(self, w2v_dir):
        print("model loading...")
        if self.test:
            model = None
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=True)
        print("model loaded.")
        return model

    def guess_from_clue(self, clue, num):
        if self.test:
            dammy_card = [(card.name, random.randint(0, 10), card.color)\
                                for card in self.field]
            sorted_card = sorted(dammy_card, key=lambda x: x[1], reverse = True)

        else:
            sorted_card = [(card.name, self.model.similarity(clue, card.name), card.color)\
                        for card in self.field]
            sorted_card = sorted(sorted_card, key=lambda x: x[1], reverse=True)

        for card in sorted_card:
            print(card)
            self.logger.info(card)

        ans_cards = sorted_card[:num]
        ans_cards_name = [card.name for card in ans_cards]
        print("answer: ", ans_cards_name)
