"""
field_logger: field card, clue and guessed word through entire turn
spymaster_logger: ranking
"""
import logging
from scipy import spatial
import numpy as np
import gensim
import csv

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

def cossim(vec1, vec2):
    """
    calculate cossine similarity
    :param vec1: np.array
    :param vec2: np.array (must be same size with vec1)
    :return: similarity (float)
    """
    return 1 - spatial.distance.cosine(vec1, vec2)

def add_noise(model, mean=0, std=0.01):
    """
    Add gaussian noise to pretrained embeddings.

    :param model: gensim model.
    :param mean:
    :param std:
    :return: dict({"word": vector})
    """
 
    new_dict = {}
    for word in model.vocab.keys():
        vector = model.wv[word]
        noise = np.random.normal(mean, std**2, len(vector))
        new_dict[word] = vector + noise
    return new_dict

def add_weighted_noise(model, mean=0, std=0.01, frequency_csv_path):
    """
    give weighted noise which depends on frequency of words.
    :param model:
    :param mean:
    :param std:
    :param frequency_data:
    :return:
    """
    new_dict = {}
    frequency_dict = {}
    if frequency_csv_path and frequency_csv_path.endwith('csv'):
        with open(frequency_csv_path, 'r') as fin:
            reader = csv.reader(fin)
            header = next(reader)
            for row in header:
                word = row[1].lower()
                freq_value = int(row[3])
                frequency_dict[word] = freq_value

    for word in model.vocab.keys():
        vector = model.wv[word]
        noise = np.random.normal(mean, std**2, len(vector))
        if word in frequency_dict:
            noise += frequency_dict[word]
        new_dict[word] = vector + noise

