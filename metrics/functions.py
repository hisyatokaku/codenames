from functools import reduce
import math
import numpy as np

def f1_score(possible_answers, actual_answers, top_n):
    """
    calculate f1 score.

    :param possible_answers: 1-d one-hot vector like: [0, 0, 1, ...]
    :param actual_answers: 1-d vector with probability like: [0.05, 0.1, 0.4, 0.03, ...]
    :return: f1_score

    threshold is set to 0 (if prob is more than 0, it is considered to be positive.
    TODO: modify this rough settings
    """

    # keep top_n values and make others zero probability
    top_n_values = sorted(actual_answers, reverse=True)[:2]
    for (i, x) in enumerate(actual_answers):
        actual_answers[i] = 0 if x not in top_n_values else s[i]

    tp, fp, fn = 0, 0, 0

    for ix in range(len(actual_answers)):
        if actual_answers[ix] > 0 and possible_answers[ix] == 0.0:
            # false positive
            fp += 1
        if actual_answers[ix] > 0 and possible_answers[ix] == 1:
            # true positive
            tp += 1
        if actual_answers[ix] == 0 and possible_answers[ix] == 1:
            # false negative
            fn += 1

    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    return 2*(precision * recall)/(precision+recall)

def cross_entropy(possible_answers, actual_answers):
    ans = 0
    for (a_label, a_prob) in zip(possible_answers, actual_answers):
        if a_prob > 0:
            ans += a_label * math.log2(a_prob)
    return ans * (-1.0)

def dcg(possible_answers, actual_answers, top_n):
    """
    calculate ndcg score.
    :param possible_answers: [0, 1, 0, 0, 1, 0,...]
    :param actual_answers: [0.01, 0.4, 0.02, 0.6, 0.01, ...]
    :param top-n: the number of elements in actual_answers you need for calculate dcg (unnecessary??)
    TODO: fix bug
    :return:
    """

    # mask the probability in actual_answers by indices of positive answer. if positive: +1, negative: -1.
    '''
    cp_actual_answers = -np.copy(actual_answers) # cast it into numpy.array
    _ = actual_answers
    least_n_indices = sorted(range(len(_)), key=lambda i: _[i])[:-top_n] # get least_n value indexes
    cp_actual_answers[least_n_indices] = 0#
    cp_actual_answers[np.nonzero(possible_answers)] *= -1
    sorted_actual_answers = -np.sort(-cp_actual_answers)
    '''
    cp_actual_answers = np.copy(actual_answers)  # cast it into numpy.array
    _ = actual_answers
    least_n_indices = sorted(range(len(_)), key=lambda i: _[i])[:-top_n]  # get least_n value indexes
    cp_actual_answers[least_n_indices] = 0  # mark least n numbers as 0 (because they are not used for dcg)
    cp_actual_answers[np.nonzero(possible_answers)] *= -1 # correct answer times -1 and it is multiplied -1 again afterward.
    sorted_actual_answers = np.sort(-cp_actual_answers)[::-1] # reversed=True

    # calculate dcg (scores are already sorted, so we do not normalize the score by calculating ndcg
    dcg = 0.
    for (ix, score) in enumerate(list(sorted_actual_answers)):
        if score > 0: # it might be good to add negative score in the sense to penalize.
            dcg += score/math.log2(ix+2) # use ix+2 to correspond to the formula's denominators which are  2, 3, 4...
    return dcg
