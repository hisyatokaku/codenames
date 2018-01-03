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
            sorted_card = [(card, self.model.similarity(clue, card.name), card.color)\
                        for card in self.field]
            sorted_card = sorted(sorted_card, key=lambda x: x[1], reverse=True)

        # limit the top
        sorted_card = sorted_card[:20]

        for card in sorted_card:
            print(card[0].name, card[1], card[2])
            self.logger.info(card)

        ans_cards = sorted_card[:num]
        print("answer: ")
        for card in sorted_card[:num]:
            print(card[0].name, card[1], card[2])

        return [card[0] for card in ans_cards]
