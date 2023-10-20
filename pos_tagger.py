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
        Transition matrix of the corpus. Contains the Log2 of the probability. log(p) of unknown = -np.inf
    """
    count = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in corpus:
        for prev_o, next_o in zip(sentence[:-1], sentence[1:]):
            count[prev_o[1]][next_o[1]] += 1

    # REVISAR EL CODIGO, ESTA AL REVES, no debe dividirse entre ese total        
    A = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for prev_o, next_posibilities in count.items():
        total = sum(next_posibilities.values())
        for next_o, freq in next_posibilities.items():
            A[prev_o][next_o] = np.log2(freq/total)
    
    return A


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
    
    vocab : set
        Vocabulary with all the training tokens.

    observations : set
        Contains all the observations found in the corpus.
    """
    count = defaultdict(lambda: defaultdict(lambda: 0))
    word_freq = defaultdict(lambda: 0)
    vocab = set(["UNK"])
    observations = set()

    for sentence in corpus:
        for word, observation in sentence:
            count[word][observation] += 1
            observations.add(observation)

    B = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for word, freqs in 

    return B, vocab, observations


def calculate_PI(corpus: list[tuple]):
    count = defaultdict(lambda: 0)
    for sentence in corpus:
        first_observation = sentence[0][1]
        count[first_observation] += 1
    
    PI = defaultdict(lambda: -np.inf)
    total = sum(count.values())
    for observation, freq in count.items():
        PI[observation] = np.log2(freq/total)
    
    return PI


def predict(sentence: list, A, B, PI, vocab)-> list:
    pass