import random
import gensim
from utils import cossim, add_noise
import pickle

class Guesser(object):
    """
    player (not spymaster) class.
    :param w2v_dir: path for pretrained embeddings
    :param field: instances of Field
    :param logger:
    """

    def __init__(self, w2v_dir, field, logger, wv_noise_pkl_path, wv_noise_value, is_wv_noise=False):
        self.w2v_dir = w2v_dir
        self.field = field
        self.logger = logger
        self.is_wv_noise = is_wv_noise # True or False
        self.wv_noise_pkl_path = wv_noise_pkl_path
        self.wv_noise_value = wv_noise_value
        self.wv = None
        self.model = self.load_model(self.w2v_dir)

    def load_model(self, w2v_dir):
        self.logger.info("player model loading...")

        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=True)
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

        self.logger.info("player model loaded.")
        return model

    def guess_from_clue(self, clue, num):
        """
        given a clue and number from spymaster, calculates all the similarity for each cards in the field.
        :param clue: word (string)
        :param num: number of cards which have to be guessed by that clue
        :return: list of the word which was guessed
        """

        if self.is_wv_noise:
            sorted_card = [(card, cossim(self.wv[clue], self.wv[card.name]), card.color)\
                    for card in self.field if card.taken_by=="None"]
            self.logger.info("score calculated by new_wv vectors.")
            sorted_card = sorted(sorted_card, key=lambda x: x[1], reverse=True)
        else:
            sorted_card = [(card, self.model.similarity(clue, card.name), card.color)\
                    for card in self.field if card.taken_by=="None"]
            sorted_card = sorted(sorted_card, key=lambda x: x[1], reverse=True)

        for card in sorted_card:
            print_text = "{} {} {}".format(card[0].name, card[1], card[2])
            self.logger.info(print_text)

        ans_cards = sorted_card

        self.logger.info("answer: ")
        for card in sorted_card[:num]:
            print_text = "{} {} {}".format(card[0].name, card[1], card[2])
            self.logger.info(print_text)

        # [(card, similarity with clue, card.color), (...), ...]
        return ans_cards