import random
import gensim
from utils import add_noise
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class Guesser(object):
    """
    player (not spymaster) class.
    :param w2v_path: path for pretrained embeddings
    :param field: instances of Field
    :param logger:
    """

    def __init__(self, field, embeddings_dict, logger):
       
        self.field = field
        self.wv = embeddings_dict
        self.logger = logger


    def guess_from_clue(self, clue):
        """
        given a clue and number from spymaster, calculates all the similarity for each cards in the field.
        :param clue: string, clue of the spumaster
        :return: full ranked list of (card, score) pairs.
        """
        sorted_card_score_pairs = []
        for card in self.field:
            if card.taken_by == "None" and clue in self.wv:
                w_vec = self.wv[clue].reshape(-1, len(self.wv[clue]))
                c_vec = self.wv[card.name].reshape(-1, len(self.wv[card.name]))
                similarity = np.asscalar(cosine_similarity(w_vec, c_vec))
                card_score_pair = (card, similarity)
            elif card.taken_by == "None" and clue not in self.wv:
                self.logger.info("clue: {} is not included in guesser embedding. similarity is set to 0.0".format(clue))
                card_score_pair = (card, 0.0)
            else:
                continue
            sorted_card_score_pairs.append(card_score_pair)
        sorted_card_score_pairs = sorted(sorted_card_score_pairs, key=lambda x: x[1], reverse=True)

        self.logger.info("Guesser ranking for all not-taken cards:")
        for card in sorted_card_score_pairs:
            print_text = "{} {} {}".format(card[0].name, card[1], card[0].color)
            self.logger.info(print_text)

        return sorted_card_score_pairs
