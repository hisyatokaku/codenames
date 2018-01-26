import random
import sys
sys.path.append('../metrics')
from functions import f1_score, cross_entropy, dcg, ndcg, codename_score
import json
from collections import defaultdict

class Card(object):
    def __init__(self, name, index):
        self.name = name
        self.color = None
        self.index = index
        self.taken_by = "None"

class Field(object):
    def __init__(self, lined_file, logger, red_metrics_path, blue_metrics_path):
        self.field = None
        self.logger = logger
        self.init_field(lined_file)
        self.red_score = 0
        self.blue_score = 0
        
        # TODO:
        # self.score = {"BLUE": 0, "RED": 0}
        
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

    def check_game_terminated(self):
        """
        check the field, then,
        if all RED or BLUE card are taken by either RED or BLUE teams,
        return True.
        otherwise, False.
        :return:
        """

        team_cards = list(filter(lambda card: card.color in ["RED", "BLUE"], self.field))
        not_taken_cards = list(filter(lambda card: card.taken_by == "None", team_cards))
        
        # Game terminated.
        if (len(not_taken_cards) == 0):
            self.game_continue = False
    
    def print_cards(self, list_of_card_score_pairs):
        """Logs a ranked list of card names and their colors to the file."""
        
        for (card, score) in list_of_card_score_pairs:
            self.logger.info("  {0:.6s}: {1}, sim={2:.2f}".format(card.color, card.name, score))
       
    def check_answer(self, team, guesser_cards): 
        """
        Checking the cards of the guesser and update points.
      
        :param team: "BLUE" or "RED", current guessing team.
        :param guesser_cards: list of (card, score) pairs.
        :return: None
        """

        guesses = []
        points = 0
        
        for card_score_pair in guesser_cards:
            card = card_score_pair[0]
            card_color = self.field[card.index].color
            self.field[card.index].taken_by = team
            
            guesses.append(card.name)

            # correct answer, plus point
            if card_color == team or card_color == "DOUBLE":
                points += 1
                
            # wrong answer, turn ends
            elif card_color == "NORMAL":
                break
                
            elif card_color == "ASSASSIN":
                self.loser = team
                self.game_continue = False   
                self.logger.info("ASSASIN discovered!")
                break

            # wrong answer, minus point, turn ends
            elif card_color != team:
                points -= 1
                break
                
            else:
                assert(False, "Untracked case in check_answer.")
        
        # TODO: store scores in dict to access them easier.
        exec("self.{}_score += {}".format(team.lower(), points))
        
        self.logger.info("\n{} team got {} points.".format(team, points))
                
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
        
        # TODO: refactor names of the arguments, e.g. to expexted_cards and predicted_cards.
        # TODO: note that answer_cards has now only clue_number elements! Check that 
        # the evaluation is still correct. Also, consider to cut possible_cards as well.

        # mask top-n cards into 1, others to 0
        # [0, 0, 1, 0, 0, 1, ...]
        self.logger.debug('evaluate_answer() running...')
        onehot_score = [0.0 for _ in range(len(self.field))]
        for (card, _ ) in possible_cards[:top_n]:
            onehot_score[card.index] += 1.
        self.logger.debug('onehot_score: ')
        self.logger.debug(onehot_score)

        # extract similarity score from answer_cards
        # [0, 0.4, 0.5, 0, 0, ...]
        # TODO: needs to be clean
        ans_score = [0.0 for _ in range(len(self.field))]
        for (card, score) in answer_cards:
            ans_score[card.index] = score

        # fieldindexed_answer_cards = sorted(answer_cards, key=lambda x: x[0].index, reverse=False)
        # ans_score = [x[1] for x in fieldindexed_answer_cards]

        self.logger.debug('ans_score: ')
        self.logger.debug(ans_score)

        # calculate value by the metrics you chose
        code_name_score = codename_score(self, team)
        f1 = f1_score(onehot_score, ans_score, top_n)
        c_e = cross_entropy(onehot_score, ans_score)
        dcg_score = dcg(onehot_score, ans_score, top_n)
        ndcg_score = ndcg(onehot_score, ans_score, top_n)
        self._update_dict(team=team, code_name_score=code_name_score, 
                          f1=f1, c_e=c_e, dcg_score=dcg_score, ndcg_score=ndcg_score)

        log_text = "f1: {}, cross_entropy: {}, dcg_score: {}".format(f1, c_e, dcg_score)
        self.logger.debug(log_text)

    def print_score(self):
        """
        print the score between red and blue.
        :return: None
        """
        self.logger.info("RED: {} vs BLUE: {}".format(self.red_score, self.blue_score))

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
