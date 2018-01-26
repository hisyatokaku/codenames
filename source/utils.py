"""
field_logger: field card, clue and guessed word through entire turn
spymaster_logger: ranking
"""
import logging
from scipy import spatial
import numpy as np

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
    load w2v from existing model, then add gaussian noise,
    restore them into pkl file.

    :param model: gensim model.
    :param mean:
    :param std:
    :return: dict({"word": vector})
    """
    vocab = model.vocab
    n_dim = len(model.wv['cat'])
    new_dict = {}
    for word in vocab.keys():
        noise = np.random.normal(mean, std**2, n_dim)
        new_wv = model.wv[word] + noise
        a_dict = {word:new_wv}
        new_dict.update(a_dict)
    return new_dict