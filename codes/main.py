from guess_from_clue import Guesser
from field import Card, Field
from give_clue import Wordrank, Spymaster
import logging
import sys
sys.path.append('../')

import argparse

# logging
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
filewriter = logging.FileHandler(filename = '../logs/spymastertest.log')
filewriter.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(filewriter)

# argparse
parser = argparse.ArgumentParser(description='input argument')
parser.add_argument('--test', '-t', type=int, default=None, help='specify test(1) or not(0)')
parser.add_argument('--cards', '-c', default='../cards/youtubetest1.txt', help='select from cards directory')
parser.add_argument('--embed', default='../models/GoogleNews.bin.gz', help='pretrained embedding')
parser.add_argument('--exname', default=None, help='experiment name.')

args = parser.parse_args()
if args.test is None:
    raise IOError('specify test or not.')
if args.test:
    print("test mode")
if not args.exname:
    print("experiment name: none")

def main():
    lined_file_path = args.cards
    w2v_path = args.embed
    is_test = args.test

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
    lined_file_path = args.cards
    w2v_path = args.embed
    is_test = args.test

    field = Field(lined_file_path, logger=logger)
    field.print_field()

    is_continue = True
    spymaster = Spymaster(w2v_path, field=field.field, logger=logger, team="RED", test=is_test)

    # print("To quit, enter Q")
    print("Spymaster set.")
    logger.info("Spymaster set.")
    # turn = True

    spymaster.give_clue(top_n = 100)

if __name__ == "__main__":
    test_spymaster()
    # main()

