import random
import gensim
from utils import cossim, add_noise
import pickle

class Guesser(object):
    """
    player (not spymaster) class.
    :param w2v_path: path for pretrained embeddings
    :param field: instances of Field
    :param logger:
    """

    def __init__(self, w2v_path, field, logger, wv_noise_pkl_path, wv_noise_value, is_wv_noise=False):
        self.w2v_path = w2v_path
        self.field = field
        self.logger = logger
        self.is_wv_noise = is_wv_noise # True or False
        self.wv_noise_pkl_path = wv_noise_pkl_path
        self.wv_noise_value = wv_noise_value
        self.wv = None
        self.model = self.load_model(self.w2v_path)

    def load_model(self, w2v_path):
        self.logger.info("Guesser model loading...")

        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True, limit=500000)
        if self.is_wv_noise:
            new_wv = add_noise(model, mean=0, std=self.wv_noise_value)
            log_text = "noise_lebel: {}".format(self.wv_noise_value)
            self.logger.info(log_text)
            self.wv = new_wv
            self.logger.info("use noised vectors.")

            # with open(self.wv_noise_pkl_path, 'wb') as w:
            #     pickle.dump(new_wv, w)
            # log_text = "noised wv saved on {}".format(self.wv_noise_pkl_path)
            # self.logger.info(log_text)

        self.logger.info("Guesser model loaded.")
        return model

    def guess_from_clue(self, clue, clue_number):
        """
        given a clue and number from spymaster, calculates all the similarity for each cards in the field.
        :param clue: string, clue of the spumaster
        :param clue_number: integer, number of cards that have to be guessed by that clue
        :return: ranked list of (card, score) pairs, guesses of the team
        """

        if self.is_wv_noise:
            sorted_card_score_pairs = [(card, cossim(self.wv[clue], self.wv[card.name]))\
                    for card in self.field if card.taken_by=="None"]
            self.logger.info("score calculated by new_wv vectors.")     
        else:
            sorted_card_score_pairs = [(card, self.model.similarity(clue, card.name))\
                    for card in self.field if card.taken_by=="None"]
        
        sorted_card_score_pairs = sorted(sorted_card_score_pairs, key=lambda x: x[1], reverse=True)

        self.logger.info("Guesser ranking, top {} will be picked:".format(clue_number))
        for card in sorted_card_score_pairs:
            print_text = "{} {} {}".format(card[0].name, card[1], card[0].color)
            self.logger.info(print_text)

        return sorted_card_score_pairs[:clue_number]
