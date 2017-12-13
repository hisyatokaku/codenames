from guess_from_clue import Field, Guesser
from give_clue import Wordrank, Spymaster
import logging
import sys

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
filewriter = logging.FileHandler(filename = '../logs/spymastertest.log')
# filewriter.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
filewriter.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(filewriter)

sys.path.append('../')

def main():
    # lined_file_path = '../cards/testset1.txt'

    lined_file_path = '../cards/youtubetest1.txt'
    w2v_path = '../models/GoogleNews.bin.gz'

    is_test = False

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
    lined_file_path = '../cards/youtubetest1.txt'
    w2v_path = '../models/GoogleNews.bin.gz'

    is_test = False

    field = Field(lined_file_path, logger=logger)
    field.print_field()

    is_continue = True
    spymaster = Spymaster(w2v_path, field=field.field, logger=logger, team="RED", test=is_test)

    # print("To quit, enter Q")
    print("Spymaster set.")
    logger.info("Spymaster set.")
    # turn = True

    spymaster.give_clue(top_n = 10)

if __name__ == "__main__":
    test_spymaster()
    # main()

