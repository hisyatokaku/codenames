import gensim
import itertools
import random
import time
import sys

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

    def give_clue(self):

        # for subsets in 2^8:
        # max(cossim(w, word in (subsets + negative))

        # suppose [c_1, c_2, ..., c_25]
        # positive_list [2, 4, 5, ..., 23]
        # others
        # give all permutation
        # for perm in permutation
        # sim(clue, perm) - sim(clue, others) for clue in models.word

        pos_ix = [card.id for card in self.field if card.color == team]
        neg_ix = [card.id for card in self.field if card.color != team]

        # make all combination
        combinations = \
            [list(comb) for n_comb in range(1, len(pos_ix)+1)\
             for comb in itertools.combinations(pos_ix, n_comb)]

        # brute force
        for comb in combinations:
            pos_card_list = [self.field[i] for i in comb]
            neg_card_list = [self.field[i] for i in neg_ix]
            for word in models.vocab:
                # want to get similarity from index

        pass
        # max(cossim(w, word in (subsets + negative))

