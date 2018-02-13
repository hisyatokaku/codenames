import random
import sys
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from metrics import compute_crossentropy, compute_dcg, compute_ndcg

class Card(object):
    def __init__(self, name, index):
        self.name = name
        self.color = None
        self.index = index
        # TODO: make in None object instead of the string.
        self.taken_by = "None"

class Field(object):
    def __init__(self, logger, metrics_path, cards_path=None, vocabulary_path=None):
        self.logger = logger
        self.metrics_path = metrics_path
        self.cards_path = cards_path
        self.vocabulary_path = vocabulary_path
        
    def reset_scores(self):   
        self.game_continue = True
        self.game_score = {"BLUE": 0, "RED": 0}
        self.metrics = {"BLUE": defaultdict(list), "RED": defaultdict(list)}
        for (i, card) in enumerate(self.field):
            self.field[i].taken_by = "None"

    def generate_cards(self):
        """
        Initialize field with cards (names and colors).
        
        If cards_path given, then the names are taken from there.
        If not, random 25 words are sampled from the vocabulary.
        
        Colors are set at random (8 for each team, 1 double, 8 normal).
        
        If there are only 5 cards in the given field, 
        it is treated as a test example and initialized with fixed colors.
        
        :param field_path: File with card names for a field, one per line.
        :param vocabulary_path: File with card names vocabulary, one per line.
        :return: None
        """
        
        if self.cards_path:
            words = open(self.cards_path, 'r').readlines()
            words = [x.rstrip().lower() for x in words]
            if len(words) != 25:
                raise ValueError("Field with incorrect number of cards given.")
        elif self.vocabulary_path:
            words = open(self.vocabulary_path, 'r').readlines()
            words = [x.strip() for x in words] # no lower() intentionally
            random.shuffle(words)  
        else:
             raise ValueError("Neither cards, nor vocabulary given for field init.")
                
        # Set card names.
        self.field = [Card(word, i) for (i, word) in enumerate(words[:25])]

        # Set colors.
        if len(self.field) == 5:
            self.init_color_for_simple_field()
        else:
            self.init_color(have_assassin=False)

        self.logger.info("New field generated:")
        self.print_field()

    def init_color_for_simple_field(self):
        """
        Set colors for toy test data of 5 cards.
        """
        
        RED_NUM = 3
        BLUE_NUM = 2
        DOUBLE_NUM = 0
        NORMAL_NUM = 0
        ASSASSIN_NUM = 0

        # set arbitrary color here
        color_ix_list = [
            0, 0, 0, 1, 1
            ]

        ix_to_str = ['RED', 'BLUE', 'DOUBLE', 'NORMAL', 'ASSASSIN']

        for (i, color_ix) in enumerate(color_ix_list):
            self.field[i].color = ix_to_str[color_ix]

    def init_color(self, have_assassin=True):
        """
        Set random colors for 25 field cards.
        
        red:0, blue:1, double:2, normal:3, assassin: 4
        """
        
        RED_NUM = 8
        BLUE_NUM = 8
        DOUBLE_NUM = 1
        NORMAL_NUM = 7
        
        if have_assassin:
            ASSASSIN_NUM = 1
        else:
            ASSASSIN_NUM = 0
            NORMAL_NUM += 1

        color_ix_list = [0 for _ in range(RED_NUM)] + \
                        [1 for _ in range(BLUE_NUM)] + \
                        [2 for _ in range(DOUBLE_NUM)] + \
                        [3 for _ in range(NORMAL_NUM)] + \
                        [4 for _ in range(ASSASSIN_NUM)]                       
        random.shuffle(color_ix_list)
    
        ix_to_str = ['RED', 'BLUE', 'DOUBLE', 'NORMAL', 'ASSASSIN']
        for (i, color_ix) in enumerate(color_ix_list):
            self.field[i].color = ix_to_str[color_ix]  

    def check_game_terminated(self):
        """
        Check wheather the *team* has just finished the game
        by guessing all remaining cards of their color. 
        """

        not_taken_cards = list(filter(lambda card: card.taken_by == "None", self.field))
        
        if (len(list(filter(lambda card: card.color in ["BLUE", "DOUBLE"], not_taken_cards))) == 0 or
            len(list(filter(lambda card: card.color in ["RED",  "DOUBLE"], not_taken_cards))) == 0):   
            self.game_continue = False
            
            self.logger.info("\nGame terminated with the score:")
            self.print_score()
            self.logger.info("-------------------------------------\n\n")
           
    def print_cards(self, list_of_card_score_pairs):
        """Logs a ranked list of card names and their colors to the file."""
        
        for (card, score) in list_of_card_score_pairs:
            card_is_taken = card.taken_by in ["BLUE", "RED"]
            self.logger.info("  {0:.6s}: {1}, sim={2:.2f}, taken={3}".format(
                    card.color, card.name, score, card_is_taken))
       
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

            # Correct answer, plus point.
            if card_color == team or card_color == "DOUBLE":
                points += 1
                
            # Wrong answer, turn ends.
            elif card_color == "NORMAL":
                break
                
            elif card_color == "ASSASSIN":
                self.game_continue = False   
                self.logger.info("ASSASIN discovered, {} loses!".format(team))
                break

            # Wrong answer, minus point, turn ends.
            elif card_color != team:
                points -= 1
                break
                
            else:
                raise ValueError("Untracked case in check_answer.")
        
        # Update score.
        self.game_score[team] += points
        
        self.logger.info("\n{} team got {} points.".format(team, points))
                
    def evaluate_answer(self, team, expected_ranking, guesser_ranking, top_n):
        """
        Compute evaluation metrics for the answer.
        :param team: "RED" or "BLUE"
        :param expected_ranking: (card, score) pairs sorted by spymaster (true labels)
        :param guesser_ranking : (card, score) pairs sorted by guesser (precitions)
        """
        
        self.logger.debug('evaluate_answer() running...')
        
        # Binary vector, 1 for expected cards.
        labels = [0 for _ in range(len(self.field))]
        for (card, _ ) in expected_ranking[:top_n]:
            labels[card.index] = 1
            
        # Binary vector, 1 for guesser cards.
        predictions = [0 for _ in range(len(self.field))]
        for (card, _) in guesser_ranking[:top_n]:
            predictions[card.index] = 1
            
        # Float vector with card probabilities from guesser.
        probabilities = [0.0 for _ in range(len(self.field))]
        for (card, score) in guesser_ranking:
            probabilities[card.index] = score
        
        # Classification measures.
        f1 = f1_score(labels, predictions)
        roc_auc = roc_auc_score(labels, probabilities)     
        crossentropy = compute_crossentropy(labels, probabilities)
        
        # Ranking measures are computed at k equal to field size! 
        sorted_rank, dcg = compute_dcg(labels, probabilities, len(labels))
        ndcg = compute_ndcg(labels, probabilities, len(labels))
        
        self._update_metrics(team=team, f1=f1, roc_auc=roc_auc, 
                             crossentropy=crossentropy, 
                             dcg=dcg, ndcg=ndcg)
     
        self.logger.debug('labels: ')
        self.logger.debug(labels)
        self.logger.debug('probabilities: ')
        self.logger.debug(probabilities)
        self.logger.debug('sorted_rank: ')
        self.logger.debug(sorted_rank)
        self.logger.debug("f1: {}, crossentropy: {}, dcg: {}, ndcg: {}".format(
                f1, crossentropy, dcg, ndcg))
 
    def _update_metrics(self, team, **kwargs):
        for key, val in kwargs.items():
            self.metrics[team][key].append(val)

    def append_game_metrics(self, multiple_game_metrics):
        for team in ["RED", "BLUE"]:
            # Average metrics over game turns.
            one_team_metrics = {key: np.mean(values_by_turns) 
                                for key, values_by_turns in self.metrics[team].items()}
            # Track only the final score.
            one_team_metrics.update({"game_score": self.game_score[team]}) 

            # Append to the given accumulated structure.
            for key, value in one_team_metrics.items():
                print("multiple: {}".format(multiple_game_metrics))
                print("one: {}".format(one_team_metrics))
                multiple_game_metrics[team][key].append(value)
        return multiple_game_metrics
      
    def dump_external_metrics(self, external_metrics):
        with open(self.metrics_path, 'a') as fout:
            json.dump(external_metrics, fout)
            fout.write("\n")
                
    def print_score(self):
        """
        print the score between red and blue.
        :return: None
        """
        self.logger.info("RED: {} vs BLUE: {}".format(self.game_score["RED"], self.game_score["BLUE"]))

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