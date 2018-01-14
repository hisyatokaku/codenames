import random
import sys
sys.path.append('../metrics')
from functions import f1_score, cross_entropy, dcg, ndcg, codename_score
import json
from collections import defaultdict

class Card(object):
    def __init__(self, name, id):
        self.name = name
        self.color = None
        self.id = id
        self.taken_by = "None"

class Field(object):
    def __init__(self, lined_file, logger, red_metrics_path, blue_metrics_path):
        self.field = None
        self.logger = logger
        self.init_field(lined_file)
        self.red_score = 0
        self.blue_score = 0
        self.game_continue = True
        self.loser = None
        self.red_metrics = defaultdict(list)
        self.blue_metrics = defaultdict(list)
        self.red_metrics_path = red_metrics_path
        self.blue_metrics_path = blue_metrics_path

    def init_field(self, lined_file):
        """
        initialize field with color and card name.
        if there are only 5 cards, it is regarded as test
        :param lined_file:
        :return: None
        """

        lines = open(lined_file, 'r').readlines()
        lines = [line.rstrip().lower() for line in lines]
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

    def _check_ending(self):
        """
        check the field, then,
        if all RED or BLUE card are taken by either RED or BLUE teams,
        return True.
        otherwise, False.
        :return:
        """
        is_end = True

        for card in self.field:
            if (card.color in ["RED", "BLUE"]):
                if (card.taken_by in ["RED", "BLUE"]):
                    pass
                else:  # card.taken_by = None, in this case.
                    # there still remains untaken card, so the game is not ended.
                    log_text = "is_end gets False.\ncard.name:{}, card.color:{}, card.taken_by:{}".format(card.name, card.color, card.taken_by)
                    self.logger.debug(log_text)
                    is_end = False
        return is_end

    def check_answer(self, team, answer_cards, top_n):
        """
        checking teams to which team the guessed cards belong.
        :input: [card, similarity with clue, card.color]
        :param team: the guessing team who is in current turn
        :param answer_cards: list of instances of Card
        :return: None
        """

        # if there is no RED card or BLUE card to be taken, game ends.
        if self._check_ending():
            self.logger.info("game terminating...")
            self.game_continue = False
            return None

        for card in answer_cards[:top_n]:
            card = card[0]
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
            # updated on 22, Jan. -> minus ally's score
            elif self.field[card.id].color == "RED" and team == "BLUE":
                exec("self.{}_score -= 1".format("BLUE".lower()))
                self.logger.info("Wrong! team: BLUE lost 1 points.")

            elif self.field[card.id].color == "BLUE" and team == "RED":
                exec("self.{}_score -= 1".format("RED".lower()))
                self.logger.info("Wrong! team: RED lost 1 points.")

            # wrong answer, turn ends
            elif self.field[card.id].color == "NORMAL":
                self.logger.info("Wrong! Normal card. {} turn ends.".format(team))

            elif self.field[card.id].color == "ASSASSIN":
                self.loser = team
                self.game_continue = False
                self.logger.info("ASSASSIN! team: {} loses.".format(team))
                break

    def evaluate_answer(self, team, possible_cards, answer_cards, top_n):
        """
        additional function for calculating score by using metrics
        memo: can it be decorator for check_answer?
        :param team:
        :param possible_cards: the cards which spymaster want player to guess
        [[card, similarity with clue], [...], ...] (sorted by similarity)

        :param answer_cards: the cards which player guessed
        [(card, similarity with clue, card.color), (...), ...] (sorted by similarity)
        # note that, len(answer_cards) might be less than 25 after round 1.
        # because from the field we strip the card which has already guessed by players.

        :return: score (type:float)
        """

        if self.red_score >= 9 or self.blue_score >= 9:
            self.logger.info("game terminated.")
            self.game_continue = False
            return None

        # mask top-n cards into 1, others to 0
        # [0, 0, 1, 0, 0, 1, ...]
        self.logger.debug('evaluate_answer() running...')
        onehot_score = [0.0 for _ in range(len(self.field))]
        for (card, _ ) in possible_cards[:top_n]:
            onehot_score[card.id] += 1.
        self.logger.debug('onehot_score: ')
        self.logger.debug(onehot_score)

        # extract similarity score from answer_cards
        # [0, 0.4, 0.5, 0, 0, ...]
        # TODO: needs to be clean
        ans_score = [0.0 for _ in range(len(self.field))]
        for (card, score, color) in answer_cards:
            ans_score[card.id] = score

        # fieldindexed_answer_cards = sorted(answer_cards, key=lambda x: x[0].id, reverse=False)
        # ans_score = [x[1] for x in fieldindexed_answer_cards]

        self.logger.debug('ans_score: ')
        self.logger.debug(ans_score)

        # calculate value by the metrics you chose
        code_name_score = codename_score(self.field, team)
        f1 = f1_score(onehot_score, ans_score, top_n)
        c_e = cross_entropy(onehot_score, ans_score)
        dcg_score = dcg(onehot_score, ans_score, top_n)
        ndcg_score = ndcg(onehot_score, ans_score, top_n)
        self._update_dict(team=team, code_name_score=code_name_score, f1=f1, c_e=c_e, dcg_score=dcg_score, ndcg_score=ndcg_score)

        log_text = "f1: {}, cross_entropy: {}, dcg_score: {}".format(f1, c_e, dcg_score)
        self.logger.debug(log_text)

    def print_score(self):
        """
        print the score between red and blue.
        :return: None
        """
        self.logger.info("RED: {} vs BLUE: {}".format(self.red_score,self.blue_score))
        # self.print_field(display_colors=True,display_taken_by=True)

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
        self.logger.info("\n")

    def _update_dict(self, team, **kwargs):
        """
        :param kwargs: metrics
        :return:
        """
        if team == "RED":
            for key, val in kwargs.items():
                self.red_metrics[key].append(val)

        if team == "BLUE":
            for key, val in kwargs.items():
                self.blue_metrics[key].append(val)

    def dump_metrics(self):
        with open(self.red_metrics_path, 'w') as w:
            json.dump(self.red_metrics, w)

        with open(self.blue_metrics_path, 'w') as w:
            json.dump(self.blue_metrics, w)
