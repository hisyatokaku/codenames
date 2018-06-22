"""
field_logger: field card, clue and guessed word through entire turn
spymaster_logger: ranking
"""
import logging
from scipy import spatial
import numpy as np
import gensim
import csv
import re
import math

def setup_filelogger(logger_name, file_name, level, add_console=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    handler = logging.FileHandler(file_name)

    if add_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    logger.addHandler(handler)
    return logger

def load_embeddings(w2v_path, logger, limit=500000):
    """
    Load pretrained embeddings.

    :param w2v_path: path to the pretrained embeddings.
    :return: gensim embeddings model.
    """

    logger.info("Loading embeddings with limits {}...".format(limit))
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True, limit=limit)
    logger.info("Embeddings loaded.")
    return embeddings

def add_noise(model, vocab_path, mode, noise_param, mean=0, freq_path=None):
    """
    Add gaussian noise to pretrained embeddings.
    :param model: gensim model.
    :param mean:
    :param std:
    :return: dict({"word": vector})
    """
    if mode not in ['n_sqrt', 'std_n_sqrt', 'random', 'n_sqrt_uniform', 'n_sqrt_ave_weight']:
        raise ValueError("guesser_noise_weight should be either one of ['n_sqrt', 'std_n_sqrt', 'random'].")

    vocab_dict = {}
    with open(vocab_path, 'r') as fin:
        for line in fin:
            word = line.strip()
            if word not in vocab_dict:
                vocab_dict[word] = 1
    # no noise for ['and/or', 'bacteria', 'loch ness', "n't", "o'clock", 'scuba diver']

    new_dict = {}

    if mode == 'n_sqrt' or mode == 'std_n_sqrt' or mode == 'n_sqrt_uniform' or mode == 'n_sqrt_ave_weight':

        if freq_path == None:
            raise ValueError("Specify the frequency path.")

        freq_dict = create_word_freq_dict(freq_path)

        for word in vocab_dict.keys():
            if word not in model.vocab:
                continue

            if word.lower() not in freq_dict:
                freq = 1
                print("word: {} freq set as 1.".format(word))
            else:
                freq = freq_dict[word.lower()]

            freq = freq if freq > 0 else 1

            vector = model.wv[word]
            if mode == 'n_sqrt':
                # noise = np.random.normal(mean, noise_param / math.sqrt(freq), len(vector))
                noise = np.random.normal(mean, noise_param, len(vector)) / math.sqrt(freq)
                # noise /= math.sqrt(freq)
            if mode == 'n_sqrt_ave_weight':
                noise = np.random.normal(mean, noise_param ** 2, len(vector))
                ave = np.mean(np.array(list(freq_dict.values())))
                noise /= math.sqrt(ave)
            if mode == 'std_n_sqrt':
                min_sqrt_std = math.sqrt(min(freq_dict.values()))
                if min_sqrt_std < 1:
                    min_sqrt_std = 1
                noise = np.random.normal(mean, noise_param * min_sqrt_std/math.sqrt(freq), len(vector))
            if mode == 'n_sqrt_uniform':
                max_freq = max(freq_dict.values())
                noise = np.random.normal(mean, noise_param ** 2, len(vector))
                noise /= math.sqrt(max_freq)
            new_dict[word] = vector + noise

    if mode == 'random':
        for word in vocab_dict.keys():
            # pass the OOV
            if word not in model.vocab:
                continue
            vector = model.wv[word]
            noise = np.random.normal(mean, noise_param ** 2, len(vector))
            new_dict[word] = vector + noise

    return new_dict

def create_word_freq_dict(freq_path):
    raw_data = open(freq_path, 'r').readlines()

    f = lambda string: re.sub('[^a-zA-Z]', '', string)
    frequency_dict = {}

    for data in raw_data:
        word, pos, cologne, freq, ra, disp = data.strip().split('\t')
        line = [word, pos, freq, ra, disp]

        word = f(line[0].lower())
        freq = int(line[2])

        if word not in frequency_dict:
            frequency_dict[word] = freq
        else:
            frequency_dict[word] = max(frequency_dict[word], freq)

    # only one field card which is not in frequency list
    frequency_dict['himalayas'] = frequency_dict['himalaya']

    return frequency_dict
    
