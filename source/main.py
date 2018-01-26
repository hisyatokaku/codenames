from guess_from_clue import Guesser
from field import Card, Field
from give_clue import Wordrank, Spymaster
from utils import setup_filelogger

import logging
import sys
import os
import argparse
import configparser

from datetime import datetime

sys.path.append('../')

# argparse
parser = argparse.ArgumentParser(description='input argument')
parser.add_argument('--setting', '-s', type=str, default=None, help='section name in ../config/settings.exp')
parser.add_argument('--exname', '-e', default=None, help='experiment name.')
parser.add_argument('--noise', '-n', type=float, default=0., help='noise for the embeddings of player. you should '
                                                                  'include this value in args.exname for readability')
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

# blue_team_logger = setup_filelogger(logger_name='blue',
#                                    file_name=os.path.join(log_dir_path, 'blue_spy_player.log'),
#                                    level=logging.INFO,
#                                    add_console=True
#                                    )

config = configparser.ConfigParser()
config.read('../Config/settings.exp')
setting_exp = config[args.setting]

for key in setting_exp:
    log_string = "{}: {}".format(key, setting_exp[key])
    ex_setting_logger.info(log_string)

def main():
    lined_file_path = setting_exp.get('cards')
    w2v_path = setting_exp.get('embed')
    # is_test = setting_exp.getint('test')
    word_table_path = setting_exp.get('spywtable')
    word_rank_list_path = setting_exp.get('spywrlist')
    spymaster_vocabulary_path = setting_exp.get('spymaster_vocabulary_path')
    wv_noise_pkl_path = setting_exp.get('wv_noise_path')
    enable_wv_noise = setting_exp.getint('enable_wv_noise')

    # hard coded paths
    red_metrics_path = os.path.join(log_dir_path, "red_metrics.json")
    blue_metrics_path = os.path.join(log_dir_path, "blue_metrics.json")

    if enable_wv_noise:
        # if there exists cmd line argument, overwrite the value
        wv_noise_value = args.noise
        if wv_noise_value == 0:
            raise ValueError("wv_noise cant be 0 with setting enable_wv_noise=1 at the same time.")
        log_text = "wv_noise_value set: {}".format(wv_noise_value)
        ex_setting_logger.info(log_text)
    else:
        wv_noise_value = -1

    field = Field(lined_file_path, logger=field_logger,
                  red_metrics_path=red_metrics_path, blue_metrics_path=blue_metrics_path)
    field.print_field()

    
    guesser = Guesser(w2v_path, field=field.field, logger=team_logger,
                      wv_noise_pkl_path=wv_noise_pkl_path,
                      wv_noise_value=wv_noise_value,
                      is_wv_noise=enable_wv_noise)
    
    spymaster = Spymaster(w2v_path, field=field.field,
                          logger=team_logger,
                          word_table_path=word_table_path,
                          word_rank_list_path=word_rank_list_path,
                          vocabulary_path=spymaster_vocabulary_path)

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
    
    field_logger.info("\nGame terminated with the score:")
    field.print_score()

    
if __name__ == "__main__":
    main()