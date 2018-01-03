from guess_from_clue import Guesser
from field import Card, Field
from give_clue import Wordrank, Spymaster
import logging
import sys

sys.path.append('../')

import argparse
import configparser

# logging
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
filewriter = logging.FileHandler(filename='../logs/spymastertest.log')
filewriter.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(filewriter)

# argparse
parser = argparse.ArgumentParser(description='input argument')
parser.add_argument('--setting', '-s', type=str, default=None, help='section name in ../config/settings.exp')
parser.add_argument('--exname', default=None, help='experiment name.')

args = parser.parse_args()

config = configparser.ConfigParser()
config.read('../Config/settings.exp')
setting_exp = config[args.setting]

for key in setting_exp:
    print(key, ": ", setting_exp[key])

if not args.exname:
    print("experiment name: none")

def main():
    lined_file_path = setting_exp.get('cards')
    w2v_path = setting_exp.get('embed')
    is_test = setting_exp.getint('test')
    word_table_path = setting_exp.get('spywtable')
    word_rank_list_path = setting_exp.get('spywrlist')
    restrict_words_path = setting_exp.get('spyrwords')

    field = Field(lined_file_path, logger=logger)
    field.print_field()

    red_guesser = Guesser(w2v_path, field=field.field, logger=logger, test=is_test)
    red_spymaster = Spymaster(w2v_path, field=field.field,
                              logger=logger, team="RED",
                              word_table_path=word_table_path,
                              word_rank_list_path=word_rank_list_path,
                              restrict_words_path=restrict_words_path,
                              test=is_test)

    blue_guesser = Guesser(w2v_path, field=field.field, logger=logger, test=is_test)
    blue_spymaster = Spymaster(w2v_path, field=field.field,
                               logger=logger, team="BLUE",
                               word_table_path=word_table_path,
                               word_rank_list_path=word_rank_list_path,
                               restrict_words_path=restrict_words_path,
                               test=is_test)

    # print("To quit, enter Q")
    logger.info("To quit, enter Q.")
    turn = True
    turn_count = 0

    while field.game_continue:
        if turn:
            cur_team = "RED"
            print("turn: ", 'RED')
            logger.info("turn: RED")

            clue, num = red_spymaster.give_clue(turn=cur_team+str(turn_count), top_n=100)
            answers = red_guesser.guess_from_clue(clue, num)  # [Card0, Card1, ...]
            field.check_answer(team=cur_team, answer_cards=answers)

        else:
            cur_team = "BLUE"
            print("turn: ", 'BLUE')
            logger.info("turn: BLUE")

            clue, num = blue_spymaster.give_clue(turn=cur_team+str(turn_count), top_n=100)
            answers = blue_guesser.guess_from_clue(clue, num)  # [Card0, Card1, ...]
            field.check_answer(team=cur_team, answer_cards=answers)

        field.print_score()
        turn = not turn
        turn_count += 1

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


def test_spymaster():
    lined_file_path = setting_exp.get('cards')
    w2v_path = setting_exp.get('embed')
    is_test = setting_exp.getint('test')
    word_table_path = setting_exp.get('spywtable')
    word_rank_list_path = setting_exp.get('spywrlist')
    restrict_words_path = setting_exp.get('spyrwords')

    field = Field(lined_file_path, logger=logger)
    field.print_field()

    spymaster = Spymaster(w2v_path, field=field.field,
                          logger=logger, team="RED",
                          word_table_path=word_table_path,
                          word_rank_list_path=word_rank_list_path,
                          restrict_words_path=restrict_words_path,
                          test=is_test)

    # print("To quit, enter Q")
    print("Spymaster set.")
    logger.info("Spymaster set.")
    # turn = True
    spymaster.give_clue(top_n=100)


if __name__ == "__main__":
    # test_spymaster()
    main()
