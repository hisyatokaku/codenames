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

    def __init__(self, field, embeddings_dict, logger):
       
        self.field = field
        self.wv = embeddings_dict
        self.logger = logger
       

    def guess_from_clue(self, clue, clue_number):
        """
        given a clue and number from spymaster, calculates all the similarity for each cards in the field.
        :param clue: string, clue of the spumaster
        :param clue_number: integer, number of cards that have to be guessed by that clue
        :return: ranked list of (card, score) pairs, guesses of the team
        """
    
        sorted_card_score_pairs = [(card, cossim(self.wv[clue], self.wv[card.name]))\
                                   for card in self.field if card.taken_by=="None"]
        sorted_card_score_pairs = sorted(sorted_card_score_pairs, key=lambda x: x[1], reverse=True)

        self.logger.info("Guesser ranking, top {} will be picked:".format(clue_number))
        for card in sorted_card_score_pairs:
            print_text = "{} {} {}".format(card[0].name, card[1], card[0].color)
            self.logger.info(print_text)

        return sorted_card_score_pairs[:clue_number]
