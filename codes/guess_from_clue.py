import random
import gensim
import time
import sys

class Card(object):
    def __init__(self, name, id):
        self.name = name
        self.color = None
        self.id = id

class Field(object):
    
    def __init__(self, lined_file, logger):
        self.field = None
        self.logger = logger
        lines = open(lined_file, 'r').readlines()
        lines = [line.rstrip().lower() for line in lines]
        self.init_field(lines)
        
    def init_field(self, lines):
        self.field = [Card(word, i) for (i, word) in enumerate(lines)]
        self.init_color()
        # print("field set.")
        self.logger.info("field set.")
        
    def init_color(self):
        """
        red:0, blue:1, double:2, normal:3, assassin: 4
        """
        RED_NUM = 8
        BLUE_NUM = 8
        DOUBLE_NUM = 1
        NORMAL_NUM = 7
        ASSASSIN_NUM = 1

        color_ix_list = [0 for _ in range(RED_NUM)] +\
                        [1 for _ in range(BLUE_NUM)] +\
                     [2 for _ in range(DOUBLE_NUM)] +\
                     [3 for _ in range(NORMAL_NUM)] +\
                     [4]
        random.shuffle(color_ix_list)

        # set arbitrary color here
        color_ix_list = [
            0, 3, 0, 3, 3,\
            0, 3, 1, 1, 3,\
            0, 0, 0, 3, 1,\
            1, 1, 1, 4, 0,\
            3, 1, 0, 1, 0\
        ]
        
        ix_to_str = ['RED', 'BLUE', 'DOUBLE', 'NORMAL', 'ASSASSIN']
        
        for (i, color_ix) in enumerate(color_ix_list):
            self.field[i].color = ix_to_str[color_ix]

    def print_field(self, display_colors = True):
        maxwordlen = max([len(card.name) for card in self.field])
        print_string = ""
        for (i, card) in enumerate(self.field):
            print_string += card.name.rjust(maxwordlen+2)
            # print(card.name.rjust(maxwordlen+2),  end='', flush=True),
            # logger.info(card.name.rjust(maxwordlen+2))
            if (i+1)%5 == 0:
                print(print_string)
                self.logger.info(print_string)
                print_string = ""

        print("\n")
        self.logger.info("\n")

        if display_colors:
            for (i, card) in enumerate(self.field):
                print_string += card.color.rjust(maxwordlen+2)
                # logger.info(card.color.rjust(maxwordlen+2))
                if (i+1)%5 == 0:
                    print(print_string)
                    # print("\n", end='', flush=True),
                    self.logger.info(print_string)
                    print_string = ""

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

    def guess_from_clue(self, clue):
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

