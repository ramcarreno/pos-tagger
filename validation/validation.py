import numpy as np
from functools import reduce

def predict_all(corpus, tagger):
    corpus_p = []
    for sentence in test:
        s = reduce(lambda x, y: x + ' ' + y, map(lambda x: x[0], sentence))
        _, s_p, _ = tagger.viterbi_best_path(s)
        corpus_p.append(s_p)

    return corpus_p


def prob_of_error_propagation(expected_list, prediction_list):
    """
    This function measures the probability of if a token prediction is wrong
    wich is the probability that the next token prediction is also wrong.
    """
    n = 0
    propagations = 0
    N = len(expected_list)
    for i in range(N):
        expected, prediction = expected_list[i], prediction_list[i]
        mask = list(map(lambda x: x[0]!=x[1], zip(expected, prediction)))
        for prev, act in zip(mask[:-1], mask[1:]):
            if prev:
                n += 1
                if act:
                    propagations += 1
    
    if n!=0:
        return propagations/n
    else:
        return -1


def get_confussion_matrix(corpus, corpus_p, tagset):
    N = len(tagset)
    cm = np.zeros((N, N))
    for i in range(len(corpus)):
        expected, prediction = corpus[i], corpus_p[i]
        for token, token_p in zip(map(lambda x: x[1], expected), map(lambda x: x[1], prediction)):
            cm[tagset.index(token), tagset.index(token_p)] += 1
    
    return cm