import sys
import numpy as np

from src.tagger import get_transition_matrix, get_emission_matrix, get_initial_state
from src.scrapper import parse_conllu_file

sys.path.insert(1, "..")  # TODO: change this


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
