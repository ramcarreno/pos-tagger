from collections import defaultdict
import numpy as np
from typing import List

from src.scrapper import parse_conllu_file


def get_transition_matrix(corpus: List[List[tuple]]):
    """
    Compute a transition matrix from a corpus of tagged sentences.

    Parameters
    ----------
    corpus : list[list[tuple]]
        A list of sentences, where each sentence is represented as a list of
        (word, POS_tag) tuples.

    Returns
    -------
    transition_matrix : defaultdict[defaultdict]
        A matrix representing the probabilities of transitioning from one POS tag to another,
        containing the log2 of the probability, so that A[i][j] = log2(p(i->j)).
        A[prev_tag][current_tag] contains the log probability of transitioning from prev_tag
        to current_tag. So if we want to know the log-probability of seeing a "NOUN" after a
        "DET", A["DET"]["NOUN"] should be accessed.
    """
    # count all instances of each distinct tag following each distinct tag
    count = defaultdict(lambda: defaultdict(lambda: 0))
    for sentence in corpus:
        for prev, current in zip(sentence[:-1], sentence[1:]):
            count[prev[1]][current[1]] += 1

    # calculate the probability each distinct tag follows each distinct tag
    # and store in the form of a transition_matrix
    transition_matrix = defaultdict(lambda: defaultdict(lambda: -np.inf))  # probability 0 to logprob -> -inf
    for prev_tag, possible_tags in count.items():
        total = sum(possible_tags.values())
        for current_tag, freq in possible_tags.items():
            transition_matrix[prev_tag][current_tag] = np.log2(freq / total)

    return transition_matrix


def get_emission_matrix(corpus: List[List[tuple]], unk_threshold=3):
    """
    Compute the emission probabilities from a corpus of tagged sentences.

    Parameters
    ----------
    corpus : list[list[tuple]]
        A list of sentences, where each sentence is represented as a list of
        (word, POS_tag) tuples.

    unk_threshold : int
        An integer representing the threshold to determine if a word should be considered
        unknown with respect to the corpus. If the word occurs less than or equal to the
        specified threshold, it will be labeled as 'UNK'.

    Returns
    -------
    tuple[defaultdict[defaultdict], set, set]
        A tuple containing three elements:

        emission_matrix : defaultdict[defaultdict]
            Emission probabilities matrix of the given corpus, containing the log2 of each
            probability.

        vocab : set
            Vocabulary (words) within all the training tokens.

        tags : set
            Contains all the possible states (distinct tags) found in the corpus.
    """
    # init accumulators
    count = defaultdict(lambda: defaultdict(lambda: 0))
    words = defaultdict(lambda: 0)
    vocab = {"UNK"}
    tags = set()

    # count all words appearing in corpus + save all possible tags
    for sentence in corpus:
        for word, tag in sentence:
            words[word] += 1
            tags.add(tag)

    # filter vocabulary from words taking into account 'unk_threshold'
    for word, freq in words.items():
        if freq > unk_threshold:
            vocab.add(word)

    # count all distinct tags associated to each word
    for sentence in corpus:
        for word, tag in sentence:
            if word in vocab:
                count[word][tag] += 1
            else:
                count["UNK"][tag] += 1

    # calculate the probability each distinct word is categorized as each distinct tag
    # and store in the form of an emission_matrix
    emission_matrix = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for word, possible_tags in count.items():
        total = sum(possible_tags.values())
        for tag, freq in possible_tags.items():
            emission_matrix[word][tag] = np.log2(freq / total)

    return emission_matrix, vocab, tags


def get_initial_state(corpus: List[List[tuple]]):  # TODO: refactor
    """
    Calculates the probability of being in the start of a sentence every possible state.

    Parameters
    ----------
    corpus : list[list[tuple]]
        All the training senteces as a list of tuples.

    Returns
    -------
    PI : ????
        ...
    """
    count = defaultdict(lambda: 0)
    for sentence in corpus:
        first_state = sentence[0][1]
        count[first_state] += 1

    PI = defaultdict(lambda: -np.inf)
    total = sum(count.values())
    for state, freq in count.items():
        PI[state] = np.log2(freq / total)

    return PI


def viterbi_logprobs(  # TODO: refactor
    A: defaultdict,
    B: defaultdict,
    PI: defaultdict,
    sentence: str,
    states: list,
    vocab: set,
):
    """
    Parameters
    ----------
    A: transition matrix -> NxN where N is the number of states that can occur (e.g. Noun, Verbs, Adjectives, etc)
    B: observation matrix (probability that a word belongs to state) -> NxT where T is the length of the sentence.
    PI: initial probabilities matrix (probability of a word of being in the beginning of the sentence 1xN.

    Returns
    -------
    viterbi: np.array
        matrix with the probabilities computed at each step t
    backpointer: list[int]
        list of indexes of the optimal
    best_logprobability: float
        probability of the best path found

    """
    # variable initialization
    words = sentence.split(" ")
    words = [word if word in vocab else "UNK" for word in words]

    N = len(states)
    T = len(words)
    viterbi = np.full((N, T), -np.inf)
    backpointer = []

    # initialization step
    for idx, state in enumerate(states):
        viterbi[idx, 0] = PI[state] + B[words[0]][state]
    best_arg = np.argmax(viterbi[:, 0])
    best_logprob = viterbi[best_arg, 0]
    backpointer.append(best_arg)

    for idx_word, word in enumerate(words[1:], 1):  # skipping first word
        for idx_state, state in enumerate(states):
            viterbi[idx_state, idx_word] = (
                best_logprob + A[states[backpointer[-1]]][state] + B[word][state]
            )
        best_arg = np.argmax(viterbi[:, idx_word])
        best_logprob = viterbi[best_arg, idx_word]
        backpointer.append(best_arg)

    return viterbi, backpointer, best_logprob
