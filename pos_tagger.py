from collections import defaultdict
import numpy as np


def calculate_A(corpus: list[list[tuple]]):
    """
    Calculates the transition matrix, the corpus vocabulary, and the observations.

    Parameters
    ----------
    corpus : list[list[tuple]]
        All the training senteces as a list of tuples.

    Returns
    -------
    A : defaultdict[defaultdict]
        Transition matrix of the corpus. Contains the Log2 of the probability.

    vocab : set
        Vocabulary with all the training tokens.

    observations : set
        Contains all the observations found in the corpus.
    """
    count = defaultdict(lambda: defaultdict(lambda: 0))
    vocab = set(["UNK"])
    observations = set()

    for sentence in corpus:
        vocab.add(sentence[0][0])
        observations.add(sentence[0][1])
        for prev_o, next_o in zip(sentence[:-1], sentence[1:]):
            count[prev_o[1]][next_o[1]] += 1
            vocab.add(next_o[0])
            observations.add(next_o[1])

    A = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for prev_o, next_os in count.items():
        total = sum(next_os.values())
        for next_o, freq in next_os.items():
            A[prev_o][next_o] = np.log2(freq / total)

    return A, vocab, observations


def calculate_B(corpus: list[tuple], unk_threshold=3):
    """
    Takes a corpus as a list of tuples and returns the emission probabilities.

    Parameters
    ----------
    corpus : list[tuple]
        All the training senteces as a list of tuples.
    unk_threshold : int
        Threshold to consider if a word is UNK or not.

    Returns
    -------
    B : ????
        Matrix...
    """
    pass


def calculate_PI(corpus: list[tuple], unk_threshold=3):
    pass


def predict(sentence: list, A, B, PI, vocab) -> list:
    pass


def viterbi_logprobs(
    A: np.array, B: np.array, PI: np.array
) -> tuple[np.array, list, float]:
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
    N = A.shape[0]
    T = B.shape[1]
    viterbi = np.full((N, T), -np.inf)
    backpointer = []

    # initialization step
    viterbi[:, 0] = PI[0] + B[:, 0]
    best_arg = np.argmax(viterbi[:, 0])
    backpointer.append(best_arg)

    for i in range(1, T):
        viterbi[:, i] = viterbi[backpointer[-1], i - 1] + A[backpointer[-1]] + B[:, i]
        best_arg = np.argmax(viterbi[:, i])
        backpointer.append(best_arg)

    best_logprobability = np.max(viterbi[:, -1])

    return viterbi, backpointer, best_logprobability
