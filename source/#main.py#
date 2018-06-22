import logging
import sys
import os
import argparse
import configparser
import numpy as np
import pickle

from datetime import datetime
from collections import defaultdict

from utils import *
from field import Field
from guess_from_clue import Guesser
from give_clue import Spymaster

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

# model directory
model_dir_path = os.path.join('../models/', datestring)
if args.exname:
    model_dir_path = model_dir_path + "_" + str(args.exname)
if not os.path.exists(model_dir_path):
    os.mkdir(model_dir_path)

table_path = os.path.join(model_dir_path, "spymaster_table")
if not os.path.exists(table_path):
    os.mkdir(table_path)

noise_dict_path = os.path.join(model_dir_path, "guesser_noise_dict")
if not os.path.exists(noise_dict_path):
    os.mkdir(noise_dict_path)

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


def play_one_game_diff_strategy(field, red_embeddings_spymaster, blue_embeddings_spymaster,
                                red_noised_embeddings_dict, blue_noised_embeddings_dict,
                                spymaster_vocabulary_path, keep_similarities_table,
                                delta, penalize_negative, normalize_negative, alpha, field_logger, team_logger,
                                wv_noise_std, game_count):
    """
    RED: try to give the clue with largest similarity with field cards
    BLUE: try to use threshold to pick clue which has as many similar fields as it can

    """
    red_similarities_table_path = os.path.join(table_path, "{}_game{}.pkl".format("RED", game_count))
    blue_similarities_table_path = os.path.join(table_path, "{}_game{}.pkl".format("BLUE", game_count))

    red_spymaster = Spymaster(field=field.field,
                              team="RED",
                          embeddings=red_embeddings_spymaster,
                          vocabulary_path=spymaster_vocabulary_path,
                          keep_similarities_table=keep_similarities_table,
                          similarities_table_path=red_similarities_table_path,
                          logger=team_logger,
                          game_count=game_count)

    red_guesser = Guesser(field=field.field, embeddings_dict=red_noised_embeddings_dict, logger=team_logger)

    blue_spymaster = Spymaster(field=field.field,
                               team="BLUE",
                          embeddings=blue_embeddings_spymaster,
                          vocabulary_path=spymaster_vocabulary_path,
                          keep_similarities_table=keep_similarities_table,
                          similarities_table_path=blue_similarities_table_path,
                          logger=team_logger,
                          game_count=game_count)

    blue_guesser = Guesser(field=field.field, embeddings_dict=blue_noised_embeddings_dict, logger=team_logger)

    red_noise_path = os.path.join(noise_dict_path, "red_noise_dict_{}".format(wv_noise_std))
    blue_noise_path = os.path.join(noise_dict_path, "blue_noise_dict_{}".format(wv_noise_std))

    with open(red_noise_path, 'wb') as red:
        pickle.dump(red_noised_embeddings_dict, red)
    with open(blue_noise_path, 'wb') as blue:
        pickle.dump(blue_noised_embeddings_dict, blue)

    team = "RED"
    turn = True
    turn_count = 0

    while field.game_continue:
        if turn:
            team = "RED"
            spymaster, guesser = red_spymaster, red_guesser
        else:
            team = "BLUE"
            spymaster, guesser = blue_spymaster, blue_guesser

        field_logger.info("\n" + "-----{} turn-----".format(team))
        team_logger.info("\n" + "-----{} turn-----".format(team))

        # Spymaster action.
        if team=="RED":
            clue_instance, is_hacky_clue = spymaster.give_one_clue(team, turn_count, top_to_print=1)
        else:
            clue_instance, is_hacky_clue = spymaster.give_clue_with_threshold(team, turn_count, delta,
                                                                              penalize_negative, normalize_negative,
                                                                              alpha, top_to_print=1)
        clue, clue_number, spymaster_ranking = clue_instance.clue, clue_instance.clue_number, clue_instance.sorted_card_score_pairs

        field_logger.info("\nClue given: {}:{}".format(str(clue), str(clue_number)))
        field.print_cards(spymaster_ranking[:clue_number + 2])

        # Guesser action.
        guesser_ranking = guesser.guess_from_clue(clue)
        guesser_cards = guesser_ranking[:clue_number]
        field_logger.info("\nGuesses given:")
        field.print_cards(guesser_cards)

        # Evaluation.
        field.check_answer(team=team, guesser_cards=guesser_cards)
        field.evaluate_spymaster_threshold(clue_instance)
        field.evaluate_spymaster_strategy(clue_instance.team, is_hacky_clue)
        field.evaluate_answer(team=team, expected_ranking=spymaster_ranking,
                              guesser_ranking=guesser_ranking, top_n=clue_number)
        field.check_game_terminated()
        turn = not turn
        turn_count += 1

def play_one_game(field, red_embeddings_spymaster, blue_embeddings_spymaster,
                  red_noised_embeddings_dict, blue_noised_embeddings_dict,
                  spymaster_vocabulary_path, keep_similarities_table,
                  delta, penalize_negative, normalize_negative, alpha, field_logger, team_logger,
                  wv_noise_std, game_count):

    red_similarities_table_path = os.path.join(table_path, "{}_game{}.pkl".format("RED", game_count))
    blue_similarities_table_path = os.path.join(table_path, "{}_game{}.pkl".format("BLUE", game_count))

    red_noise_path = os.path.join(noise_dict_path, "red_noise_dict_{}".format(wv_noise_std))
    blue_noise_path = os.path.join(noise_dict_path, "blue_noise_dict_{}".format(wv_noise_std))

    with open(red_noise_path, 'wb') as red:
        pickle.dump(red_noised_embeddings_dict, red)
    with open(blue_noise_path, 'wb') as blue:
        pickle.dump(blue_noised_embeddings_dict, blue)

    red_spymaster = Spymaster(field=field.field,
                              team="RED",
                              embeddings=red_embeddings_spymaster,
                              vocabulary_path=spymaster_vocabulary_path,
                              keep_similarities_table=keep_similarities_table,
                              similarities_table_path=red_similarities_table_path,
                              logger=team_logger,
                              game_count=game_count)

    blue_spymaster = Spymaster(field=field.field,
                               team="BLUE",
                               embeddings=blue_embeddings_spymaster,
                               vocabulary_path=spymaster_vocabulary_path,
                               keep_similarities_table=keep_similarities_table,
                               similarities_table_path=blue_similarities_table_path,
                               logger=team_logger,
                               game_count=game_count)

    red_guesser = Guesser(field=field.field, embeddings_dict=red_noised_embeddings_dict, logger=team_logger)
    blue_guesser = Guesser(field=field.field, embeddings_dict=blue_noised_embeddings_dict, logger=team_logger)

    team = "RED"
    turn = True
    turn_count = 0

    while field.game_continue:
        if turn:
            team = "RED"
            spymaster = red_spymaster
            guesser = red_guesser
        else:
            team = "BLUE"
            spymaster = blue_spymaster
            guesser = blue_guesser

        field_logger.info("\n" + "-----{} turn-----".format(team))
        team_logger.info( "\n" + "-----{} turn-----".format(team))

        # Spymaster action.
        clue_instance, is_hacky_clue = spymaster.give_clue_with_threshold(team, turn_count, delta, penalize_negative, normalize_negative, alpha, top_to_print=1)
        clue, clue_number, spymaster_ranking = clue_instance.clue, clue_instance.clue_number, clue_instance.sorted_card_score_pairs

        field_logger.info("\nClue given: {}:{}".format(str(clue), str(clue_number)))
        field.print_cards(spymaster_ranking[:clue_number + 2])
 
        # Guesser action.
        guesser_ranking = guesser.guess_from_clue(clue)
        guesser_cards = guesser_ranking[:clue_number]
        field_logger.info("\nGuesses given:")
        field.print_cards(guesser_cards)
        
        # Evaluation.
        field.check_answer(team=team, guesser_cards=guesser_cards)
        field.evaluate_spymaster_threshold(clue_instance)
        field.evaluate_spymaster_strategy(clue_instance.team, is_hacky_clue)
        field.evaluate_answer(team=team, expected_ranking=spymaster_ranking, 
                              guesser_ranking=guesser_ranking, top_n=clue_number)
        field.check_game_terminated()
        turn = not turn
        turn_count += 1
    
def setup_noise_params():
    # Noise for the guesser embeddings:
    std_min = setting_exp.getfloat('wv_noise_std_min')
    std_max = setting_exp.getfloat('wv_noise_std_max')
    std_step = setting_exp.getfloat('wv_noise_std_step')
    
    # No noise, if it's not specified
    if std_min is None or std_max is None or std_step is None:
        std_min = 0
        std_max = 0
        std_step = 0.1
    
    # Set range including the max. If min=0, max=0, then the list is [0]:
    wv_noise_std_range = np.arange(std_min, std_max + std_step, std_step)

    if std_step == 0:
        wv_noise_std_range = np.array([std_min])

    if len(wv_noise_std_range) < 1:
        raise ValueError("Noise range has no elements. Check if max is greater than min.")

    # if std_max > 1:
    #     raise ValueError("std_max is too large.")

    ex_setting_logger.info("Range for gaussian noise std: {}".format(wv_noise_std_range))
    
    return wv_noise_std_range

def setup_noise_scaler():
    scaler_min = setting_exp.getfloat('wv_noise_scaler_min')
    scaler_max = setting_exp.getfloat('wv_noise_scaler_max')
    scaler_step = setting_exp.getfloat('wv_noise_scaler_step')

    if scaler_min is None or scaler_max is None or scaler_step is None:
        scaler_min = 1
        scaler_max = 1
        scaler_step = 1

    scaler_std_range = np.arange(scaler_min, scaler_max + scaler_step, scaler_step)

    # scaler_std_range = np.geomspace(scaler_min, scaler_max, scaler_step)

    if len(scaler_std_range) < 1 or scaler_max < 1:
        raise ValueError("Scaler range has no elements . Check if max is greater than min.\n or scaler_max is too small.")

    ex_setting_logger.info("Range for scaler for std: {}".format(scaler_std_range))

    return scaler_std_range

def setup_averaging():
    games_averaging = setting_exp.getint('games_averaging') 
    if not games_averaging:
        games_averaging = 1
    return games_averaging

def setup_threshold_delta():
    delta_min = setting_exp.getfloat('delta_min')
    delta_max = setting_exp.getfloat('delta_max')
    delta_step = setting_exp.getfloat('delta_step')

    # No delta, if it's not specified
    if delta_min is None or delta_max is None or delta_step is None:
        delta_min, delta_max, delta_step = 0, 0, 0.1
    if delta_step == 0:
        delta_range = np.array([delta_min])
    else:
        delta_range = np.arange(delta_min, delta_max + delta_step, delta_step)

    if len(delta_range) < 1:
        raise ValueError("Delta range has no elements. Check if max is greater than min.")
    ex_setting_logger.info("Range for delta range: {}".format(delta_range))
    return delta_range

def setup_alpha():
    alpha_min = setting_exp.getfloat('alpha_min')
    alpha_max = setting_exp.getfloat('alpha_max')
    alpha_step = setting_exp.getfloat('alpha_step')

    # No alpha, if it's not specified
    if alpha_min is None or alpha_max is None or alpha_step is None:
        alpha_min, alpha_max, alpha_step = 0, 0, 0.1
    alpha_range = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)
    if len(alpha_range) < 1:
        raise ValueError("Alpha range has no elements. Check if max is greater than min.")
    ex_setting_logger.info("Range for alpha range: {}".format(alpha_range))
    return alpha_range

def setup_embeddings():
    red_embeddings_spymaster_path = setting_exp.get('red_embeddings_spymaster_path')
    red_embeddings_guesser_path = setting_exp.get('red_embeddings_guesser_path')
    red_embeddings_path = setting_exp.get('red_embeddings_path')

    blue_embeddings_spymaster_path = setting_exp.get('blue_embeddings_spymaster_path')
    blue_embeddings_guesser_path = setting_exp.get('blue_embeddings_guesser_path')
    blue_embeddings_path = setting_exp.get('blue_embeddings_path')

    is_all_embeddings_spy_guess_set = (red_embeddings_spymaster_path and red_embeddings_guesser_path and
                                       blue_embeddings_spymaster_path and blue_embeddings_guesser_path) 
    is_any_embeddings_spy_guess_set = (red_embeddings_spymaster_path or red_embeddings_guesser_path or
                                       blue_embeddings_spymaster_path or blue_embeddings_guesser_path)
    use_common_embeddings_btw_spy_guess = red_embeddings_path and blue_embeddings_path
    use_any_commmon_embeddings_btw_spy_guess = (red_embeddings_path or blue_embeddings_path) and (not use_common_embeddings_btw_spy_guess)
    
    if use_common_embeddings_btw_spy_guess:
        if is_any_embeddings_spy_guess_set:
            raise ValueError("""some {red, blue}_embeddings_{spy, guesser} are set.\n To use same embeddings
            between spy and guesser, remove this from config.""")

        ex_setting_logger.info("""using same embeddings across spy and guesser.\n
                               red_embeddings: {}\nblue_embeddings: {}
                               """.format(red_embeddings_path, blue_embeddings_path))
        red_embeddings = load_embeddings(red_embeddings_path, ex_setting_logger, limit=500000)
        blue_embeddings = load_embeddings(blue_embeddings_path, ex_setting_logger, limit=500000)

        red_embeddings_spymaster, red_embeddings_guesser = red_embeddings, red_embeddings
        blue_embeddings_spymaster, blue_embeddings_guesser = blue_embeddings, blue_embeddings

    elif is_all_embeddings_spy_guess_set:
        red_embeddings_spymaster = load_embeddings(red_embeddings_spymaster_path, ex_setting_logger, limit=500000)
        red_embeddings_guesser = load_embeddings(red_embeddings_guesser_path, ex_setting_logger, limit=500000)
        blue_embeddings_spymaster = load_embeddings(blue_embeddings_spymaster_path, ex_setting_logger, limit=500000)
        blue_embeddings_guesser = load_embeddings(blue_embeddings_guesser_path, ex_setting_logger, limit=500000)
        
        if use_any_commmon_embeddings_btw_spy_guess:
            raise ValueError("either red or blue embeddings are unset")
         
    # some of the parameters unset.
    else:
        raise ValueError("either red or blue embeddings are unset, check your config file.")

    return red_embeddings_spymaster, red_embeddings_guesser, blue_embeddings_spymaster, blue_embeddings_guesser

def _setup_embeddings():
    """
    DEPRECATED
    if both embeddings_path_spymaster and embeddings_path_guesser are found, load them separately,
    if only embeddings_path are found, load it and share them between spymaster and guesser.
    """


    embeddings_path = setting_exp.get('embeddings_path')
    embeddings_path_spymaster = setting_exp.get('embeddings_path_spymaster')
    embeddings_path_guesser = setting_exp.get('embeddings_path_guesser')

    # Load pretrained embeddings.
    if embeddings_path_spymaster and embeddings_path_guesser:
        # load spymaster and guesser embeddings
        embeddings_spymaster = load_embeddings(embeddings_path_spymaster, ex_setting_logger, limit=500000)
        embeddings_guesser = load_embeddings(embeddings_path_guesser, ex_setting_logger, limit=500000)

        # if 3 embeddings are found, raise an error.
        if embeddings_path:
            raise ValueError("extra embeddings found. Remove either embeddings_path or embeddings_path_{epymaster, guesser}.")

    # either one of them is empty, or two of them are empty.
    else:
        if embeddings_path:
            embeddings = load_embeddings(embeddings_path, ex_setting_logger, limit=500000)
            embeddings_spymaster, embeddings_guesser = embeddings, embeddings

        elif embeddings_path_spymaster or embeddings_path_guesser:
            raise ValueError("Both of spymaster embeddings and guesser embeddings must be designated.")
        else:
            raise ValueError("no embeddings found.")

    return embeddings_spymaster, embeddings_guesser

def main():
    cards_path = setting_exp.get('cards')
    field_vocabulary_path = setting_exp.get('field_vocabulary_path')
    spymaster_vocabulary_path = setting_exp.get('spymaster_vocabulary_path')
    # use_diff_embeddings = setting_exp.get('use_different_embeddings')
    keep_similarities_table = setting_exp.getint('keep_similarities_table')
    metrics_path = os.path.join(log_dir_path, "metrics.json")
    penalize_negative = setting_exp.getint('add_penalize_negative_for_clue_score')
    normalize_negative = setting_exp.getint('normalize_negative_for_score')
    diff_strategy = setting_exp.getint('different_strategy')
    guesser_noise_weight = setting_exp.get('guesser_noise_weight')
    freq_path = setting_exp.get('frequency_path')

    games_averaging = setup_averaging()
    wv_noise_std_range = setup_noise_params()
    wv_noise_scaler = setup_noise_scaler()

    if guesser_noise_weight == 'std_n_sqrt':
        noise_range = wv_noise_scaler
    else:
        noise_range = wv_noise_std_range
    threshold_delta_range = setup_threshold_delta()
    alpha_range = setup_alpha()

    red_embeddings_spymaster, red_embeddings_guesser, blue_embeddings_spymaster, blue_embeddings_guesser = setup_embeddings()

    # Create field, not initialized with cards yet.
    field = Field(field_logger, metrics_path, vocabulary_path=field_vocabulary_path)

    for game_count in range(games_averaging):
        # generate the field.
        field.generate_cards()
        for noise_param in noise_range:
            # prepare dict for guesser
            # guesser shares same vocab with spymaster
            red_noised_embeddings_dict = add_noise(model=red_embeddings_guesser, vocab_path=spymaster_vocabulary_path,
                                                   mode=guesser_noise_weight, noise_param=noise_param, mean=0,
                                                   freq_path=freq_path) # seems to be super slow, TBD
            blue_noised_embeddings_dict = add_noise(model=blue_embeddings_guesser, vocab_path=spymaster_vocabulary_path,
                                                    mode=guesser_noise_weight, noise_param=noise_param, mean=0,
                                                    freq_path=freq_path) # seems to be super slow, TBD

            field.logger.info("------Noise param set to {0:.2f}.-------\n".format(noise_param))

            for threshold_delta in threshold_delta_range:
                field.logger.info("------Delta set to {0:.3f}.-------\n".format(threshold_delta))
                # For each noise level, play multiple games on different fields.
                for alpha in alpha_range:
                    field.reset_scores()

                    field.logger.info("------Alpha set to {0:.3f}.-------\n".format(alpha))
                    if diff_strategy:
                        field.logger.info("playing greedy vs only one...")
                        play_one_game_diff_strategy(field, red_embeddings_spymaster, blue_embeddings_spymaster,
                        red_noised_embeddings_dict, blue_noised_embeddings_dict, spymaster_vocabulary_path,
                        keep_similarities_table, threshold_delta, penalize_negative,
                        normalize_negative, alpha, field_logger, team_logger,
                        noise_param, game_count)
                    
                    else:
                        play_one_game(field, red_embeddings_spymaster, blue_embeddings_spymaster,
                        red_noised_embeddings_dict, blue_noised_embeddings_dict, spymaster_vocabulary_path,
                        keep_similarities_table, threshold_delta, penalize_negative,
                        normalize_negative, alpha, field_logger, team_logger,
                        noise_param, game_count)

                    # Play games with different noise level.
                    multiple_game_metrics = {"RED": defaultdict(list), "BLUE": defaultdict(list)}
                    hparams = {"hparams_alpha": alpha,
                               "hparams_delta": threshold_delta,
                               "hparams_noise": noise_param,
                               "game_count": game_count+1
                               }
                    for key, val in hparams.items():
                        multiple_game_metrics.update({key: val})
                    multiple_game_metrics = field.append_game_metrics(multiple_game_metrics)
                    # Dump metrics of the games along with the hparams.
                    field.dump_external_metrics(multiple_game_metrics)

if __name__ == "__main__":
    main()
