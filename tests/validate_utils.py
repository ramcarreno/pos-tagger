import sys
import numpy as np

from src.pos_tagger import get_transition_matrix, get_emission_matrix
from src.scrapper import parse_conllu_file

sys.path.insert(1, "..")  # TODO: change this


def validate_transition_matrix(transition_matrix, epsilon=0.0000001):
    for tag in transition_matrix.keys():
        total = 0
        for log_p in transition_matrix[tag].values():
            total += np.exp2(log_p)
        if abs(total - 1) > epsilon:
            raise Exception(
                f"ERROR: {tag}. Sum of probabilities {total} should be 1."
            )


def validate_emission_matrix(emission_matrix, epsilon=0.0000001):
    for word in emission_matrix.keys():
        total = 0
        for log_p in emission_matrix[word].values():
            total += np.exp2(log_p)

        if abs(total - 1) > epsilon:
            raise Exception(
                f"ERROR: {word}. Sum of probabilities {total} should be 1."
            )


def validate_initial_state(PI, epsilon=0.0000001):
    total = 0
    for log_p in PI.values():
        total += np.exp2(log_p)

    if abs(total - 1) > epsilon:
        raise Exception(f"ERROR. Sum of probabilities {total} should be 1.")
