from collections import defaultdict
import numpy as np


def calculate_A(corpus: list[list[tuple]]):
    """
    Takes a corpus as a list, each element of the list is a different sentence, and each line contains a list of tuples. Calculates the transition matrix, and the vocabulary.
    
    Parameters
    ----------
    corpus : list[tuple]
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
            A[prev_o][next_o] = np.log2(freq/total)
    
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


def predict(sentence: list, A, B, PI, vocab)-> list:
    pass