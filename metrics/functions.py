from functools import reduce
import math
import numpy as np

def softmax(_list):
    """
    :param _list: type:list
    :return: type:list
    """
    return list(np.exp(_list)/np.sum(np.exp(_list),axis=0))

def f1_score(gt_label, prediction, top_n):
    """
    calculate f1 score.

    :param gt_label: 1-d one-hot vector like: [0, 0, 1, ...]
    :param prediction: 1-d vector with probability like: [0.05, 0.1, 0.4, 0.03, ...]
    :return: f1_score
    """

    # keep top_n values and make others zero probability
    top_n_values = sorted(prediction, reverse=True)[:top_n]
    for (i, x) in enumerate(prediction):
        prediction[i] = 0 if x not in top_n_values else s[i]

    tp, fp, fn = 0, 0, 0

    for ix in range(len(prediction)):
        if prediction[ix] > 0 and gt_label[ix] == 0.0:
            # false positive
            fp += 1
        if prediction[ix] > 0 and gt_label[ix] == 1:
            # true positive
            tp += 1
        if prediction[ix] == 0 and gt_label[ix] == 1:
            # false negative
            fn += 1

    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    return 2*(precision * recall)/(precision+recall)

def cross_entropy(gt_label, prediction):
    """
    calculate cross_entropy score.
    :param gt_label: [0, 1, 0, 0, 1, 0,...]
    :param prediction: [0.01, 0.4, 0.02, 0.6, 0.01, ...]
    :return: cross_entropy (float)
    """
    sf_prediction = softmax(prediction)
    return sum([l * p for(l, p) in zip(gt_label, sf_prediction)])
    # return reduce(lambda su,(x,y): su+x*y, zip(gt_label, sf_prediction), 0)

def dcg(gt_label, prediction, top_n):
    """
    calculate dcg score.
    :param gt_label: [0, 1, 0, 0, 1, 0,...]
    :param prediction: [0.01, 0.4, 0.02, 0.6, 0.01, ...]
    :param top-n: the number of elements in prediction you need for calculate dcg (unnecessary??)
    :return:
    """

    # calculate the rank of the element in prediction array. This should be rank used in the dcg formula.
    sorted_rank_index = list(np.array(prediction).argsort()[::-1].argsort()+1)
    return sum([rel/math.log2(rank+1) for (rel, rank) in zip(gt_label, sorted_rank_index)])

def ndcg(gt_label, prediction, top_n):
    """
    calculate ndcg score.
    :param gt_label: [0, 1, 0, 0, 1, 0,...]
    :param prediction: [0.01, 0.4, 0.02, 0.6, 0.01, ...]
    :param top-n: the number of elements in prediction you need for calculate dcg (unnecessary??)
    :return:
    """
    ideal_rank_index = list(np.array(gt_label).argsort()[::-1].argsort() + 1)
    DCG = dcg(gt_label, prediction, top_n)
    IDCG = sum([rel / math.log2(rank+1) for (rel, rank) in zip(gt_label, ideal_rank_index)])
    return DCG/IDCG

