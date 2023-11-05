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


def get_error_propagation_prob(expected_list, prediction_list):
    """
    This function measures the probability of if a token prediction is wrong
    which is the probability that the next token prediction is also wrong.
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
        # There are no errors so we cant measure the probability
        return -1


def get_accuracy(confusion_matrix):
    # get the matrix diagonal and sum all the correct predictions (true positives for each tag)
    diagonal = np.diagonal(confusion_matrix)
    total_correct = np.sum(diagonal)

    # sum the total number of predictions made
    total_predictions = np.sum(confusion_matrix)

    # accuracy is defined as the ratio of correct predictions made
    accuracy = total_correct / total_predictions

    return accuracy


def get_precision(confusion_matrix):
    # number of tags
    total_tags = len(confusion_matrix)

    precisions, predictions = [], []

    for i in range(total_tags):
        # each column contains all the info we need
        total_tag_correct = confusion_matrix[i, i]  # tp for that tag
        total_tag_predictions = np.sum(confusion_matrix[:, i])

        # precision is defined as the ratio of true predictions from all positive (applies to a certain tag)
        if total_tag_predictions == 0:  # some tags may not appear in the test set at all
            precision = 0
        else:
            precision = total_tag_correct / total_tag_predictions

        precisions.append(precision)
        predictions.append(int(total_tag_predictions))

    # micro precision (weighted avg)
    diagonal = np.diagonal(confusion_matrix)
    micro_precision = np.dot(precisions, diagonal) / sum(diagonal)

    # macro precision (plain avg of all classes, not weighted)
    macro_precision = np.mean(precisions)

    return precisions, predictions, micro_precision, macro_precision


def get_recall(confusion_matrix):
    # number of tags
    total_tags = len(confusion_matrix)

    recalls, predictions = [], []

    for i in range(total_tags):
        # each column contains all the info we need
        total_tag_correct = confusion_matrix[i, i]  # tp for that tag
        total_tag_predictions = np.sum(confusion_matrix[i, :])

        # recall is defined as the true positive ratio (applies to a certain tag)
        if total_tag_predictions == 0:  # some tags may not appear in the test set at all
            recall = 0
        else:
            recall = total_tag_correct / total_tag_predictions
        recalls.append(recall)
        predictions.append(int(total_tag_predictions))

    # micro recall (weighted avg)
    diagonal = np.diagonal(confusion_matrix)
    micro_recall = np.dot(recalls, diagonal) / sum(diagonal)

    # macro recall (plain avg of all classes, not weighted)
    macro_recall = np.mean(recalls)

    return recalls, predictions, micro_recall, macro_recall


def get_f1(precision, recall):
    return 2 / (1 / precision + 1 / recall)