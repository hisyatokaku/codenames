from guess_from_clue import Guesser
from field import Card, Field
from give_clue import Spymaster
from utils import *

import logging
import sys
import os
import argparse
import configparser
import numpy as np

from datetime import datetime

sys.path.append('../')

# argparse
parser = argparse.ArgumentParser(description='input argument')
parser.add_argument('--setting', '-s', type=str, default=None, help='section name in ../config/settings.exp')
parser.add_argument('--exname', '-e', default=None, help='experiment name.')
args = parser.parse_args()
print(args.exname)

# log directory
datestring = (datetime.now().strftime('%m%d%Y_%H%M'))
log_dir_path = os.path.join('../logs/', datestring)
if args.exname:
     log_dir_path = log_dir_path + "_" + str(args.exname)

if not os.path.exists(log_dir_path):
    os.mkdir(log_dir_path)

# logging config, setup
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

# loggers
ex_setting_logger = setup_filelogger(logger_name='experiment_setting',
                                     file_name=os.path.join(log_dir_path, 'exp_setting.log'),
                                     level=logging.INFO,
                                     add_console=False)

field_logger = setup_filelogger(logger_name='field',
                                file_name=os.path.join(log_dir_path, 'field.log'),
                                level=logging.INFO,
                                add_console=False)

team_logger = setup_filelogger(logger_name='team',
                               file_name=os.path.join(log_dir_path, 'team.log'),
                               level=logging.INFO,
                               add_console=False)

config = configparser.ConfigParser()
config.read('../config/settings.exp')
setting_exp = config[args.setting]

for key in setting_exp:
    log_string = "{}: {}".format(key, setting_exp[key])
    ex_setting_logger.info(log_string)


def play_one_game(field, spymaster_embeddings, guesser_embeddings_dict,
                  spymaster_vocabulary_path, similarities_table_path, 
                  field_logger, team_logger):
    
    spymaster = Spymaster(field=field.field,
                          embeddings=spymaster_embeddings,
                          vocabulary_path=spymaster_vocabulary_path,
                          similarities_table_path=similarities_table_path,
                          logger=team_logger)
    
    guesser = Guesser(field=field.field, embeddings_dict=guesser_embeddings_dict, logger=team_logger)
  
    team = "RED"
    turn = True
    turn_count = 0

    while field.game_continue:
        if turn:
            team = "RED"
        else:
            team = "BLUE"
                  
        field_logger.info("\n" + "-----{} turn-----".format(team))
        team_logger.info( "\n" + "-----{} turn-----".format(team))

        # Spymaster action.
        clue, clue_number, spymaster_ranking = spymaster.give_clue_with_threshold(team, turn_count, top_n=10)
    
        field_logger.info("\nClue given: {}:{}".format(str(clue), str(clue_number)))
        field.print_cards(spymaster_ranking[:clue_number + 2])
 
        # Guesser action.
        guesser_cards = guesser.guess_from_clue(clue, clue_number)
        
        field_logger.info("\nGuesses given:")
        field.print_cards(guesser_cards)
        
        field.check_answer(team=team, guesser_cards=guesser_cards)
        field.evaluate_answer(team=team, possible_cards=spymaster_ranking, answer_cards=guesser_cards, top_n=clue_number)
        field.check_game_terminated()
        
        turn = not turn
        turn_count += 1

    field.dump_metrics()
    field_logger.info("-------------------------------------\n\n")
    
    
def main():
    cards_path = setting_exp.get('cards')
    embeddings_path = setting_exp.get('embeddings_path')
    field_vocabulary_path = setting_exp.get('field_vocabulary_path')
    spymaster_vocabulary_path = setting_exp.get('spymaster_vocabulary_path')
    similarities_table_path = setting_exp.get('similarities_table')
    
    # Noise for the guesser embeddings:
    std_min = setting_exp.getfloat('wv_noise_std_min')
    std_max = setting_exp.getfloat('wv_noise_std_max')
    std_step = setting_exp.getfloat('wv_noise_std_step')
    # Including the max. If min=0, max=0, then the list is [0]:
    wv_noise_std_range = np.arange(std_min, std_max + std_step, std_step)     
    if len(wv_noise_std_range) < 1:
        raise ValueError("Noise range has no elements. Check if max is greater than min.")
    ex_setting_logger.info("Range for gaussian noise std: {}".format(wv_noise_std_range))

    # Hard coded paths for metrics:
    red_metrics_path = os.path.join(log_dir_path, "red_metrics.json")
    blue_metrics_path = os.path.join(log_dir_path, "blue_metrics.json")
    
    # Load pretrained embeddings.
    embeddings = load_embeddings(embeddings_path, ex_setting_logger, limit=500000)
    
    # TBD: average across several fields.
    field = Field(logger=field_logger, 
                  red_metrics_path=red_metrics_path, blue_metrics_path=blue_metrics_path,
                  cards_path=cards_path, vocabulary_path=field_vocabulary_path)
    
    # Play games with different noise level.
    for vw_noise_std in wv_noise_std_range:   
        field.reset_scores()
        noised_embeddings_dict = add_noise(model=embeddings, mean=0, std=vw_noise_std) # seems to be super slow, TBD
        field.logger.info("Noise std set to {}.".format(vw_noise_std)) 
        
        play_one_game(field, embeddings, noised_embeddings_dict,
                      spymaster_vocabulary_path, similarities_table_path, 
                      field_logger, team_logger)

    
if __name__ == "__main__":
    main()