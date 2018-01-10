import random

class Card(object):
    def __init__(self, name, id):
        self.name = name
        self.color = None
        self.id = id
        self.taken_by = "None"

class Field(object):
    def __init__(self, lined_file, logger):
        self.field = None
        self.logger = logger
        lines = open(lined_file, 'r').readlines()
        lines = [line.rstrip().lower() for line in lines]
        self.init_field(lines)

        self.red_score = 0
        self.blue_score = 0
        self.game_continue = True
        self.loser = None

    def init_field(self, lines):
        """
        initialize field with color and card name.
        if there are only 5 cards, it is regarded as test
        :param lines:
        :return: None
        """
        self.field = [Card(word, i) for (i, word) in enumerate(lines)]
        if len(self.field) == 5:
            self.init_color_for_simple_field()
        else:
            self.init_color()

        self.logger.info("field set.")

    def init_color_for_simple_field(self):
        """
        initialize field with color and card name.
        :return: None
        """
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

    def check_answer(self, team, answer_cards):
        """
        checking teams to which team the guessed cards belong.
        :param team: the guessing team who is in current turn
        :param answer_cards: list of instances of Card
        :return: None
        """
        if self.red_score > 9 or self.blue_score > 9:
            self.logger.info("game terminated.")
            self.game_continue = False

        for card in answer_cards:
            log_text = "guessed card: {}".format(str(card.name))
            self.logger.info(log_text)
            self.field[card.id].taken_by = team

            # correct answer
            if self.field[card.id].color == team:
                exec("self.{}_score += 1".format(team.lower()))
                self.logger.info("Correct! team: {} got 1 points.".format(team))

            elif self.field[card.id].color == "DOUBLE":
                exec("self.{}_score += 1".format(team.lower()))
                self.logger.info("Correct! team: {} got 1 points.".format(team))

            # wrong answer, give score to enemy, turn ends
            elif self.field[card.id].color == "RED" and team == "BLUE":
                exec("self.{}_score += 1".format("RED".lower()))
                self.logger.info("Wrong! team: RED got 1 points. BLUE turn ends.")
                break
            elif self.field[card.id].color == "BLUE" and team == "RED":
                exec("self.{}_score += 1".format("BLUE".lower()))
                self.logger.info("Wrong! team: BLUE got 1 points. RED turn ends.")
                break

            # wrong answer, turn ends
            elif self.field[card.id].color == "NORMAL":
                self.logger.info("Wrong! Normal card. {} turn ends.".format(team))
                break
            elif self.field[card.id].color == "ASSASSIN":
                self.loser = team
                self.game_continue = False
                self.logger.info("ASSASSIN! team: {} loses.".format(team))
                break

    def print_score(self):
        """
        print the score between red and blue.
        :return: None
        """

        self.logger.info("RED: {} vs BLUE: {}".format(self.red_score,self.blue_score))
        self.print_field(display_colors=True,display_taken_by=True)

    def print_field(self, display_colors=True, display_taken_by=False):
        maxwordlen = max([len(card.name) for card in self.field])
        print_string = ""
        for (i, card) in enumerate(self.field):
            print_string += card.name.rjust(maxwordlen + 2)
            if (i + 1) % 5 == 0:
                self.logger.info(print_string)
                print_string = ""
        self.logger.info("\n")

        if display_colors:
            print_string = ""
            for (i, card) in enumerate(self.field):
                print_string += card.color.rjust(maxwordlen + 2)
                if (i + 1) % 5 == 0:
                    self.logger.info(print_string)
                    print_string = ""
        self.logger.info("\n")

        if display_taken_by:
            print_string = ""
            for (i, card) in enumerate(self.field):
                print_string += card.taken_by.rjust(maxwordlen + 2)
                if (i + 1) % 5 == 0:
                    self.logger.info(print_string)
                    print_string = ""
