import numpy as np
from functools import reduce


def validate_transition_matrix(transition_matrix, epsilon=0.0000001):
    """
    Validate the transition probabilities computed for a transition matrix.

    This function checks if the sum of probabilities for each tag in the transition matrix
    is approximately equal to 1 within a specified `epsilon` tolerance.

    Parameters
    ----------
    transition_matrix : dict
        A dictionary representing the transition probabilities between POS tags.
    epsilon : float
        The tolerance level for the sum of probabilities. Defaults to 0.0000001.

    Raises
    ------
    Exception
        If the sum of probabilities of all the tags in the transition matrix is not at least
        greater than 1 minus the specified tolerance (`epsilon`).
    """
    for tag in transition_matrix.keys():
        total = 0
        for log_p in transition_matrix[tag].values():
            total += np.exp2(log_p)

        if abs(total - 1) > epsilon:
            raise Exception(
                f"ERROR: {tag}. Sum of probabilities {total} should be 1."
            )


def validate_emission_matrix(emission_matrix, epsilon=0.0000001):
    """
    Validate the emission probabilities in an emission matrix.

    This function checks if the sum of probabilities for each word in the emission
    matrix is approximately equal to 1 within a specified `epsilon` tolerance.

    Parameters
    ----------
    emission_matrix : dict
        A dictionary representing the emission probabilities of words for POS tags.
    epsilon : float
        The tolerance level for the sum of probabilities. Defaults to 0.0000001.

    Raises
    ------
    Exception
        If the sum of probabilities of all the tags in the emission matrix is not at least
        greater than 1 minus the specified tolerance (`epsilon`).
    """
    for word in emission_matrix.keys():
        total = 0
        for log_p in emission_matrix[word].values():
            total += np.exp2(log_p)

        if abs(total - 1) > epsilon:
            raise Exception(
                f"ERROR: {word}. Sum of probabilities {total} should be 1."
            )


def validate_initial_state(initial_state, epsilon=0.0000001):
    """
    Validate the initial state probabilities.

    This function checks if the sum of initial state probabilities for all POS tags is
    approximately equal to 1 within a specified `epsilon` tolerance.

    Parameters
    ----------
    initial_state : dict
        A dictionary representing the initial state probabilities for part-of-speech tags.
    epsilon : float
        The tolerance level for the sum of probabilities. Defaults to 0.0000001.

    Raises
    ------
    Exception
        If the sum of probabilities for of all the tags in the initial state vector is not at
        least greater than 1 minus the specified tolerance (`epsilon`).
    """
    total = 0
    for log_p in initial_state.values():
        total += np.exp2(log_p)

    if abs(total - 1) > epsilon:
        raise Exception(f"ERROR: Sum of probabilities {total} should be 1.")


def predict_all(corpus, tagger):
    corpus_p = []
    for sentence in corpus:
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
        mask = list(map(lambda x: x[0] != x[1], zip(expected, prediction)))
        for prev, act in zip(mask[:-1], mask[1:]):
            if prev:
                n += 1
                if act:
                    propagations += 1

    if n != 0:
        return propagations / n
    else:
        return -1


def get_confusion_matrix(corpus, corpus_p, tagset):
    N = len(tagset)
    cm = np.zeros((N, N))
    for i in range(len(corpus)):
        expected, prediction = corpus[i], corpus_p[i]
        for token, token_p in zip(map(lambda x: x[1], expected), map(lambda x: x[1], prediction)):
            cm[tagset.index(token), tagset.index(token_p)] += 1

    return cm
