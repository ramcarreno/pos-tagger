from collections import defaultdict
import numpy as np


def calculate_A(corpus: list[list[tuple]]):
    """ 
    Calculates the transition matrix, the corpus vocabulary, and the states.
    
    Parameters
    ----------
    corpus : list[list[tuple]]
        All the training senteces as a list of tuples.

    Returns
    -------
    A : defaultdict[defaultdict]
        Transition matrix of the corpus. Contains the Log2 of the probability. log(p) of unknown = -np.inf
        This matrix works as: A[i][j] = log2(p(i->j))
        So if we want to know which is the probability of having a "NOUN" after a "DET", we should check A["DET"]["NOUN"]
    """
    count = defaultdict(lambda: defaultdict(lambda: 0))
    
    for sentence in corpus:
        for prev, actual in zip(sentence[:-1], sentence[1:]):
            count[prev[1]][actual[1]] += 1

    A = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for prev_state, act_posibilities in count.items():
        total = sum(act_posibilities.values())
        for actual_state, freq in act_posibilities.items():
            A[prev_state][actual_state] = np.log2(freq/total)
    
    return A


def test_A(A, states, epsilon=0.0000001):
    for i in states:
        suma = 0
        for j in states:
            suma += np.exp2(A[i][j])
        if abs(suma-1)>epsilon:
            print(f"ERROR: {i}. SUM: {suma} should be 1.")


# IT SHOULD BE APLIED .lower() to any token??????
def calculate_B(corpus: list[list[tuple]], unk_threshold=3):
    """
    Takes a corpus as a list of tuples and returns the emission probabilities.
    
    Parameters
    ----------
    corpus : list[list[tuple]]
        All the training senteces as a list of tuples.
    
    unk_threshold : int
        Threshold to consider if a word is UNK or not. If the number of occurences of a certain word is less or equal than the threshold, this word is going to be categorized as "UNK".

    Returns
    -------
    B : defaultdict[defaultdict]
        Emission probabilities matrix of the given corpus. Contains the Log2 of the probability. log(p) of unknown = -np.inf
    
    vocab : set
        Vocabulary with all the training tokens.

    states : set
        Contains all the states found in the corpus.
    """
    count = defaultdict(lambda: defaultdict(lambda: 0))
    word_freq = defaultdict(lambda: 0)
    vocab = set(["UNK"])
    states = set()

    for sentence in corpus:
        for word, state in sentence:
            word_freq[word] += 1
            states.add(state)
    
    for word, freq in word_freq.items():
        if freq > unk_threshold:
            vocab.add(word)
    
    for sentence in corpus:
        for word, state in sentence:
            if word in vocab:
                count[word][state] += 1
            else:
                count["UNK"][state] += 1
    
    B = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for word, posibilities in count.items():
        total = sum(posibilities.values())
        for state, freq in posibilities.items():
            B[word][state] = np.log2(freq/total)

    return B, vocab, states


def test_B(B, vocab, epsilon=0.0000001):
    for word in vocab: 
        if abs(sum(B[word].values())-1)>epsilon:
            print(f"ERROR: {word}. SUM: {abs(sum(B[word].values())-1)} should be 1.")


def calculate_PI(corpus: list[list[tuple]]):
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
        PI[state] = np.log2(freq/total)
    
    return PI
        

def predict(sentence: list, A, B, PI, vocab)-> list:
    pass