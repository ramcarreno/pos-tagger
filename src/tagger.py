from collections import defaultdict
import numpy as np
from typing import List

from scrapper import parse_conllu_file


# TODO: classes
# tagger.HiddenMarkovModelTrainer
# tagger.HiddenMarkovModelTagger

class HiddenMarkovModelTrainer:
    def __init__(self, corpus: List[List[tuple]]):
        self.corpus = corpus

    def train(self):
        return HiddenMarkovModelTagger(
            self.transition_matrix(),
            self.emission_matrix(),
            self.initial_state(),
            self.tagset(),
            self.vocabulary()
        )

    def vocabulary(self, unk_threshold=3):
        words = defaultdict(lambda: 0)
        vocab = {"UNK"}

        # count all words appearing in corpus + save all possible tags
        for sentence in self.corpus:
            for word, tag in sentence:
                words[word] += 1

        # filter vocabulary from words taking into account 'unk_threshold'
        for word, freq in words.items():
            if freq > unk_threshold:
                vocab.add(word)

        return vocab

    def tagset(self):
        # save all possible tags
        tags = set()
        for sentence in self.corpus:
            for word, tag in sentence:
                tags.add(tag)
        tags = list(tags)  # convert resulting set to list so an order is kept
        return tags

    def transition_matrix(self):
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
            containing the log2 of each probability, so that A[i][j] = log2(p(i->j)).

            A[prev_tag][current_tag] contains the log-probability of transitioning from prev_tag
            to current_tag. So if we want to know the log-probability of seeing a "NOUN" after a
            "DET", A["DET"]["NOUN"] should be accessed.
        """
        # init accumulators
        count = defaultdict(lambda: defaultdict(lambda: 0))

        # count all instances of each distinct tag following each distinct tag
        for sentence in self.corpus:
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

    def emission_matrix(self):
        """
        Compute the emission probabilities from a corpus of tagged sentences.

        Parameters
        ----------
        corpus : list[list[tuple]]
            A list of sentences, where each sentence is represented as a list of
            (word, POS_tag) tuples.

        Returns
        -------
        emission_matrix : defaultdict[defaultdict]
            A tuple containing three elements:

            emission_matrix : defaultdict[defaultdict]
                Emission probabilities matrix of the given corpus, containing the log2 of each
                probability.

            vocab : set
                Vocabulary (words) within all the training tokens.

            tags : list
                Contains all the possible states (distinct tags) found in the corpus.
        """
        # init accumulators
        count = defaultdict(lambda: defaultdict(lambda: 0))
        vocab = self.vocabulary()

        # count all distinct tags associated to each word
        for sentence in self.corpus:
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

        return emission_matrix

    def initial_state(self):
        """
        Compute the initial state (first word tag) probabilities from a corpus of tagged sentences.

        Parameters
        ----------
        corpus : list[list[tuple]]
            A list of sentences, where each sentence is represented as a list of
            (word, POS_tag) tuples.

        Returns
        -------
        initial_state : defaultdict
            A vector representing the probabilities of the first word of a sentence being a certain
            type of POS tag, containing the log2 of each probability.
        """
        # init accumulators
        count = defaultdict(lambda: 0)

        # count every instance of distinct tags for the first word of every sentence
        for sentence in self.corpus:
            first_word_tag = sentence[0][1]
            count[first_word_tag] += 1

        # calculate the probability of each distinct tag being the first tag
        initial_state = defaultdict(lambda: -np.inf)
        total = sum(count.values())
        for tag, freq in count.items():
            initial_state[tag] = np.log2(freq / total)

        return initial_state


class HiddenMarkovModelTagger:
    def __init__(self, transition_matrix, emission_matrix, initial_state, tags, vocabulary):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state
        self.tags = tags
        self.vocabulary = vocabulary


def viterbi_logprobs(  # TODO: refactor & separate training from tagging!
        transitions: defaultdict,
        emissions: defaultdict,
        initial_state: defaultdict,
        sentence: str,  # TODO...
        states: list,
        vocab: set,
):
    """
    Parameters
    ----------
    transitions: transition matrix (probability that state i is followed by state j) ->
        NxN where N is the number of possible states (tags, e.g. `noun`, `verb`, `adj`).
    emissions: observation matrix (probability that a word belongs to state) ->
        NxT where N is the number of possible states and T is the length of the sentence.
    initial_state: initial probabilities matrix (probability the first word belongs to state) ->
        1xN where N is the number of possible states.

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
        viterbi[idx, 0] = initial_state[state] + emissions[words[0]][state]
    best_arg = np.argmax(viterbi[:, 0])
    best_logprob = viterbi[best_arg, 0]
    backpointer.append(best_arg)

    for idx_word, word in enumerate(words[1:], 1):  # note that it skips the first word
        for idx_state, state in enumerate(states):
            viterbi[idx_state, idx_word] = (
                    best_logprob + transitions[states[backpointer[-1]]][state] + emissions[word][state]
            )
        best_arg = np.argmax(viterbi[:, idx_word])
        best_logprob = viterbi[best_arg, idx_word]
        backpointer.append(best_arg)

    return viterbi, backpointer, best_logprob
