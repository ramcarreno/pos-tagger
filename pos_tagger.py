from collections import defaultdict

def calculate_A(corpus: list[tuple]):
    """
    Takes a corpus as a list of tuples and returns the transition matrix, and the vocabulary.
    
    Parameters
    ----------
    corpus : list[tuple]
        All the training senteces as a list of tuples.

    Returns
    -------
    A : ????
        Matrix...

    vocab : set
        Vocabulary with all the training tokens.
    """
    pass


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