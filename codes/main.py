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
parser.add_argument('--noise', '-n', type=float, default=0., help='noise for the embeddings of player. you should includ'
                                                                  'e this value in args.exname for readability')
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
                                     add_console=True
                                     )
field_logger = setup_filelogger(logger_name='field',
                                file_name=os.path.join(log_dir_path, 'field.log'),
                                level=logging.DEBUG,
                                add_console=True
                                )
red_team_logger = setup_filelogger(logger_name='red',
                                    file_name=os.path.join(log_dir_path, 'red_spy_player.log'),
                                    level=logging.INFO,
                                    add_console=True
                                    )
blue_team_logger = setup_filelogger(logger_name='blue',
                                    file_name=os.path.join(log_dir_path, 'blue_spy_player.log'),
                                    level=logging.INFO,
                                    add_console=True
                                    )

config = configparser.ConfigParser()
config.read('../Config/settings.exp')
setting_exp = config[args.setting]

for key in setting_exp:
    log_string = "{}: {}".format(key, setting_exp[key])
    ex_setting_logger.info(log_string)

def main():
    lined_file_path = setting_exp.get('cards')
    w2v_path = setting_exp.get('embed')
    is_test = setting_exp.getint('test')
    word_table_path = setting_exp.get('spywtable')
    word_rank_list_path = setting_exp.get('spywrlist')
    restrict_words_path = setting_exp.get('spyrwords')
    wv_noise_pkl_path = setting_exp.get('wv_noise_path')
    enable_wv_noise = setting_exp.getint('enable_wv_noise')
    red_metrics_path = os.path.join(log_dir_path, "red_metrics.json")
    blue_metrics_path = os.path.join(log_dir_path, "blue_metrics.json")

    if enable_wv_noise:
        # if there exists cmd line argument, overwrite the value
        wv_noise_value = args.noise
        if wv_noise_value == 0:
            raise ValueError("wv_noise cant be 0 with setting enable_wv_noise=1 at the same time.")
        log_text = "wv_noise_value set: {}".format(wv_noise_value)
        ex_setting_logger.info(log_text)

    field = Field(lined_file_path, logger=field_logger,
                  red_metrics_path=red_metrics_path, blue_metrics_path=blue_metrics_path)
    field.print_field()

    red_guesser = Guesser(w2v_path, field=field.field, logger=red_team_logger,
                          wv_noise_pkl_path=wv_noise_pkl_path,
                          wv_noise_value = wv_noise_value,
                          is_wv_noise=enable_wv_noise, test=is_test)
    red_spymaster = Spymaster(w2v_path, field=field.field,
                              logger=red_team_logger, team="RED",
                              word_table_path=word_table_path,
                              word_rank_list_path=word_rank_list_path,
                              restrict_words_path=restrict_words_path,
                              test=is_test)

    blue_guesser = Guesser(w2v_path, field=field.field, logger=blue_team_logger,
                           wv_noise_pkl_path=wv_noise_pkl_path,
                           wv_noise_value = wv_noise_value,
                           is_wv_noise=enable_wv_noise, test=is_test)

    blue_spymaster = Spymaster(w2v_path, field=field.field,
                               logger=blue_team_logger, team="BLUE",
                               word_table_path=word_table_path,
                               word_rank_list_path=word_rank_list_path,
                               restrict_words_path=restrict_words_path,
                               test=is_test)

    turn = True
    turn_count = 0

    while field.game_continue:
        if turn:
            cur_team = "RED"
            field_logger.info("turn: {}, turn count: {}".format(cur_team, turn_count))
            clue, num, possible_answers = red_spymaster.give_clue_with_threshold(turn=cur_team+str(turn_count), top_n=10)
            log_text = "clue: {}, num: {}".format(str(clue), str(num))
            field_logger.info(log_text)
            answers = red_guesser.guess_from_clue(clue, num)  # [(card, similarity with clue, card.color), (...), ...]
            field.check_answer(team=cur_team, answer_cards=answers, top_n=num)
            field.evaluate_answer(team=cur_team, possible_cards=possible_answers, answer_cards=answers, top_n=num)

        else:
            cur_team = "BLUE"
            field_logger.info("turn: {}, turn count: {}".format(cur_team, turn_count))
            clue, num, possible_answers = blue_spymaster.give_clue_with_threshold(turn=cur_team+str(turn_count), top_n=10)
            log_text = "clue: {}, num: {}".format(str(clue), str(num))
            field_logger.info(log_text)
            answers = blue_guesser.guess_from_clue(clue, num) # [(card, similarity with clue, card.color), (...), ...]
            field.check_answer(team=cur_team, answer_cards=answers, top_n=num)
            field.evaluate_answer(team=cur_team, possible_cards=possible_answers, answer_cards=answers, top_n=num)

        field.print_score()
        turn = not turn
        turn_count += 1

    # TODO: serialize metrics into files
    field.dump_metrics()
    field_logger.info("game terminated.")

def gridsearch_noise():
    noises = []
    # for noise in noises:
    pass

'''
def _main():
    lined_file_path = setting_exp.get('cards')
    w2v_path = setting_exp.get('embed')
    is_test = setting_exp.getint('test')

    # spy settings
    word_table_path = setting_exp.get('spywtable')
    word_rank_list_path = setting_exp.get('spywrlist')
    restrict_words_path = setting_exp.get('spyrwords')

    field = Field(lined_file_path, logger=logger)
    field.print_field()

    is_continue = True
    guesser = Guesser(w2v_path, field=field.field, logger=logger, test=is_test)

    # print("To quit, enter Q")
    logger.info("To quit, enter Q.")
    turn = True

    while (is_continue):
        if turn:
            print("turn: ", 'RED')
            logger.info("turn: RED")
        else:
            print("turn: ", 'BLUE')
            logger.info("turn: BLUE")

        print("clue:")
        logger.info("clue:")
        clue = input()
        print("clue:", clue)
        # logger.info("clue:")
        logger.info(clue)
        if clue == "Q":
            break

        try:
            guesser.guess_from_clue(clue)
        except:
            print("the word is not in vocabulary.")
            logger.warn("the word is not in vocabulary.")
        finally:
            turn = not turn
'''
'''
def test_spymaster():
    lined_file_path = setting_exp.get('cards')
    w2v_path = setting_exp.get('embed')
    is_test = setting_exp.getint('test')
    word_table_path = setting_exp.get('spywtable')
    word_rank_list_path = setting_exp.get('spywrlist')
    restrict_words_path = setting_exp.get('spyrwords')

    field = Field(lined_file_path, logger=field_logger)
    field.print_field()

    spymaster = Spymaster(w2v_path, field=field.field,
                          logger=red_team_logger, team="RED",
                          word_table_path=word_table_path,
                          word_rank_list_path=word_rank_list_path,
                          restrict_words_path=restrict_words_path,
                          test=is_test)

    # print("To quit, enter Q")
    print("Spymaster set.")
    red_team_logger.info("Spymaster set.")
    # turn = True
    spymaster.give_clue(top_n=100)
'''
if __name__ == "__main__":
    main()
    # gridsearch_noise()