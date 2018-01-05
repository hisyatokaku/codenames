import random
import gensim

class Guesser(object):
    """
    player (not spymaster) class.
    :param w2v_dir: path for pretrained embeddings
    :param field: instances of Field
    :param logger:
    :param test: bool
    """

    def __init__(self, w2v_dir, field, logger, test=False):
        self.test = test
        self.w2v_dir = w2v_dir
        self.field = field
        self.logger = logger
        self.model = self.load_model(self.w2v_dir)

    def load_model(self, w2v_dir):
        self.logger.info("player model loading...")
        if self.test:
            model = None
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=True)
        self.logger.info("player model loaded.")
        return model

    def guess_from_clue(self, clue, num):
        """
        given a clue and number from spymaster, calculates all the similarity for each cards in the field.
        :param clue: word (string)
        :param num: number of cards which have to be guessed by that clue
        :return: list for the word which was guessed
gkt         """
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
            print_text = "{} {} {}".format(card[0].name, card[1], card[2])
            self.logger.info(print_text)

        ans_cards = sorted_card[:num]
        self.logger.info("answer: ")
        for card in sorted_card[:num]:
            print_text = "{} {} {}".format(card[0].name, card[1], card[2])
            self.logger.info(print_text)

        return [card[0] for card in ans_cards]
