import random

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
        if len(self.field) == 5:
            self.init_color_for_simple_field()
        else:
            self.init_color()

        self.logger.info("field set.")

    def init_color_for_simple_field(self):
        RED_NUM = 2
        BLUE_NUM = 2
        DOUBLE_NUM = 0
        NORMAL_NUM = 1
        ASSASSIN_NUM = 0

        # set arbitrary color here
        color_ix_list = [
            0, 0, 0, 1, 1
            ]

        ix_to_str = ['RED', 'BLUE', 'DOUBLE', 'NORMAL', 'ASSASSIN']

        for (i, color_ix) in enumerate(color_ix_list):
            self.field[i].color = ix_to_str[color_ix]

    def init_color(self):
        """
        red:0, blue:1, double:2, normal:3, assassin: 4
        """
        RED_NUM = 8
        BLUE_NUM = 8
        DOUBLE_NUM = 1
        NORMAL_NUM = 7
        ASSASSIN_NUM = 1

        color_ix_list = [0 for _ in range(RED_NUM)] + \
                        [1 for _ in range(BLUE_NUM)] + \
                        [2 for _ in range(DOUBLE_NUM)] + \
                        [3 for _ in range(NORMAL_NUM)] + \
                        [4]
        random.shuffle(color_ix_list)

        # set arbitrary color here
        color_ix_list = [
            0, 3, 0, 3, 3, \
            0, 3, 1, 1, 3, \
            0, 0, 0, 3, 1, \
            1, 1, 1, 4, 0, \
            3, 1, 0, 1, 0 \
            ]

        ix_to_str = ['RED', 'BLUE', 'DOUBLE', 'NORMAL', 'ASSASSIN']

        for (i, color_ix) in enumerate(color_ix_list):
            self.field[i].color = ix_to_str[color_ix]

    def print_field(self, display_colors=True):
        maxwordlen = max([len(card.name) for card in self.field])
        print_string = ""
        for (i, card) in enumerate(self.field):
            print_string += card.name.rjust(maxwordlen + 2)
            # print(card.name.rjust(maxwordlen+2),  end='', flush=True),
            # logger.info(card.name.rjust(maxwordlen+2))
            if (i + 1) % 5 == 0:
                print(print_string)
                self.logger.info(print_string)
                print_string = ""

        print("\n")
        self.logger.info("\n")

        if display_colors:
            for (i, card) in enumerate(self.field):
                print_string += card.color.rjust(maxwordlen + 2)
                # logger.info(card.color.rjust(maxwordlen+2))
                if (i + 1) % 5 == 0:
                    print(print_string)
                    # print("\n", end='', flush=True),
                    self.logger.info(print_string)
                    print_string = ""
