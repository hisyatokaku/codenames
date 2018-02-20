import gensim
import itertools
import numpy as np
import pickle
import os
import csv

from functools import reduce
import random
import time
import sys

class Clue(object):
    """ Class for a clue with its score and ranked (card, score) answer pairs."""

    def __init__(self, clue, sorted_card_score_pairs, delta, team):
        self.clue = clue
        self.sorted_card_score_pairs = sorted_card_score_pairs    
        self.team = team
        self.delta = delta
        self.total_score, self.clue_number = self._calculate_score_with_threshold()

    def _calculate_score_with_threshold(self):
        """
        Helper function.

        total_score = sum_{card in clue_answers} similarity(clue, card).
        clue_answers set is defined as all the cards with similarity greater than 
        for any negative card (wrong team, assasin, normal).
      
        :return: total_score, clue_number
        """
        # TODO: compute normalized negative score and add it to the total score.
        
        clue_number = 0
        total_score = 0
    
        for card, score in self.sorted_card_score_pairs:
            # Collect positive set until the first negative word occurrence.
            if card.color in [self.team, "DOUBLE"]:
                clue_number += 1
                total_score += score
            else:
                break
                
        return total_score, clue_number
    
    def get_summary(self):
        """
        Printing function for this class.

        :param clue: object of Clue class.
        :return: text for logging
        """
        
        text = "word: {}, total_score: {} \n".format(self.clue, self.total_score)
        for card, score in self.sorted_card_score_pairs:
            card_text = "\t card.name:{} (team:{}), similarity: {} \n".format(card.name, card.color, score)
            text += card_text
        return text

    
class Vocab(object):
    """
    Vocabulary class. Same structure is implemented in gensim.models.vocab.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Spymaster(object):
    """
    Spymaster class.
    :param field: the instance from Field class (defined in field.py)
    :param embeddings: pretrained embeddings
    :param vocabulary_path: path to csv or txt file
    :param similarities_table_path (optinal): path to pkl file with clue-card similarities
    :param logger: logger for the spymaster.
    """

    def __init__(self, field, embeddings, vocabulary_path, similarities_table_path, logger):
        
        self.similarities_table_path = similarities_table_path
        self.vocabulary_path = vocabulary_path
        self.field = field
        self.logger = logger     
        self.model = embeddings
       
        self.vocab = dict()
        self.vocab_size = 0
        self.load_vocab()

        # Initialize similarities_table by -1 and  fill with actual clue-card similarities.
        self.similarities_table = np.full([self.vocab_size, len(self.field)], -1, dtype=np.float32)
        self.fill_table()

    def update_vocab(self, word):
        """
        Helper function to update vocabulary with a new word.
        Adds the word only if it occurres in the embeddings model
        vocavulry and does not occur in the field.
        
        :param word: string, word to be added.
        :return: None.
        """
        
        if (word not in self.vocab and
            word in self.model.vocab and 
            word not in [x.name for x in self.field]):
            
            self.vocab[word] = Vocab(index=self.vocab_size)
            self.vocab_size += 1
    
    
    def load_vocab(self):
        """
        Loads vocabulary.
        
        If self.vocabulary_path not provided, loads the vocabulary from embeddings model
        (3000000 words for unfiltered word2vec model on GoogleNews).
        To filter the vocabulary, specify the path to the word list in csv or txt format.
  
        :return: None.
        """

        if self.vocabulary_path:            
            # For now, the file format is derived from the file extension.
            if self.vocabulary_path.endswith('csv'):
                self.logger.info("Filter spymaster vocabulary by csv-file: {}".format(self.vocabulary_path))
                with open(self.vocabulary_path, 'r') as fin:
                    reader = csv.reader(fin)
                    header = next(reader)
                    for row in reader:
                        word = row[1].lower()
                        self.update_vocab(word)                       
            elif self.vocabulary_path.endswith('txt'):
                self.logger.info("Filter spymaster vocabulary by txt-file: {}".format(self.vocabulary_path))
                with open(self.vocabulary_path, 'r') as fin:
                    for line in fin:
                        word = line.strip()
                        self.update_vocab(word)      
            else:
                raise ValueError("Unknown file format for filter spymaster vocabulary.") 
        else:
            self.logger.info("Load spymaster vocabulary from gensim.models.KeyedVectors.")
            self.vocab = self.model.vocab
            self.vocab_size = len(self.vocab)
                    
        self.logger.info("Spymaster vocabulary size is {}".format(self.vocab_size))
  

    def fill_table(self):
        """
        Compute the similarity for each word from the vocabulary and each card from the field
        and save in self.similarities_table. The complexity is |vocabulary| * |field_size|.
        """
        
        similarities_table_path = self.similarities_table_path

        if similarities_table_path and os.path.exists(similarities_table_path):
            self.logger.info(similarities_table_path + " exists. Loading.")
            with open(similarities_table_path, 'rb') as r:
                self.similarities_table_path = pickle.load(r)

        else:
            self.logger.info("Creating similarities_table from scratch.")
            for word in self.vocab:
                for card in self.field:
                    w_ix = self.vocab[word].index
                    c_ix = card.index

                    if word not in self.model.vocab or card.name not in self.model.vocab:
                        self.logger.warning("OOV word or card :{}, setting similarity to 0.".format(word))
                        self.similarities_table[w_ix][c_ix] = 0.0
                    else:
                        self.similarities_table[w_ix][c_ix] = self.model.similarity(word, card.name)
            
            if (similarities_table_path):    
                with open(similarities_table_path, 'wb') as w:
                    pickle.dump(self.similarities_table_path, w)

    def give_clue_with_threshold(self, team, turn_count, delta, top_to_print=5):
        """
        Give a clue, maximizig the sum of similarities to the positive words set
        and minimizing the average similarity to all negative words of the field.
        
        The positive set is determined by top-k words in the predicted ranking
        of answers for the given clue; k is the largest possible value, 
        that provides only words of the right color in the set.
        
        :param team: "RED" or "BLUE".
        :param turn_count: integer, deprecated.
        :param top_to_print: how many clue candidates should be printed.
        :return:
        """
        
        if team not in ["RED", "BLUE"]:
            raise ValueError("Team string must be RED or BLUE.")
       
        clue_candidates = []
        for clue in self.vocab:
            card_score_pairs = []
            clue_ix = self.vocab[clue].index
            for card in filter(lambda x: x.taken_by == "None", self.field):
                score = self.similarities_table[clue_ix][card.index]
                card_score_pairs.append((card, score))

            sorted_card_score_pairs = sorted(card_score_pairs, key=lambda x: x[1], reverse=True)
            clue = Clue(clue, sorted_card_score_pairs, delta, team)
            clue_candidates.append(clue)
            
        clue_candidates = sorted(clue_candidates, key=lambda x: x.total_score, reverse=True)
        clue = clue_candidates[0]
        
        for clue in clue_candidates[:top_to_print]:
            self.logger.info(clue.get_summary())

        return clue.clue, clue.clue_number, clue.sorted_card_score_pairs