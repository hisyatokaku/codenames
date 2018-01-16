from functools import reduce
import math
import numpy as np

def f1_score(possible_answers, actual_answers):
    """
    calculate f1 score.

    :param possible_answers: 1-d one-hot vector like: [0, 0, 1, ...]
    :param actual_answers: 1-d vector with probability like: [0, 0.1, 0.4, 0, ...]
    :return: f1_score


    threshold is set to 0 (if prob is more than 0, it is considered to be positive.
    TODO: modify this rough settings
    """

    p_nonzero_num = reduce(lambda x,y:x+y, possible_answers)
    r_nonzero_num = reduce(lambda x,y:x+y, map(lambda z:z>0, actual_answers))

    p_nonzero_index = [ix for (ix, val) in enumerate(possible_answers) if val > 0]
    r_nonzero_index = [ix for (ix, val) in enumerate(actual_answers) if val > 0]

    # TODO: fix below by using above
    '''
    tp =
    fp =
    fn =
    presicion = float()
    recall =
    return 2*(precision*recall)/(precision+recall)
    '''
    pass

def cross_entropy(possible_answers, actual_answers):
    ans = 0
    for (a_label, a_prob) in zip(possible_answers, actual_answers):
        if a_prob > 0:
            ans += a_label * math.log2(a_prob)
    return ans

def dcg(possible_answers, actual_answers):
    """
    calculate ndcg score.
    :param possible_answers:
    :param actual_answers:
    :return:
    """
    # mask the probability in actual_answers by indices of positive answer. if positive: +1, negative: -1.
    cp_actual_answers = -np.copy(actual_answers)
    cp_actual_answers[np.nonzero(possible_answers)] *= -1
    sorted_actual_answers = -np.sort(-cp_actual_answers)

    # calculate dcg (scores are already sorted, so we do not normalize the score by calculating ndcg
    dcg = 0.
    for (ix, score) in sorted_actual_answers:
        if score > 0: # it might be good to add negative score in the sense to penalize.
            dcg += score/math.log2(ix+2) # use ix+2 to correspond to the formula's denominators which are  2, 3, 4...
    return dcg
