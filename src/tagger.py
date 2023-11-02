from collections import defaultdict
import numpy as np
from typing import List


class HiddenMarkovModel:
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
        # init
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
        # init
        tagset = set()

        # save all possible tags found in corpus
        for sentence in self.corpus:
            for word, tag in sentence:
                tagset.add(tag)

        # convert resulting set to list so an order is kept
        tagset = list(tagset)
        return tagset

    def transition_matrix(self):
        """
        Compute a transition matrix from a corpus of tagged sentences.

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

        Returns
        -------
        emission_matrix : defaultdict[defaultdict]
            Emission probabilities matrix of the given corpus, containing the log2 of each
            probability.
        """
        # init accumulators
        count = defaultdict(lambda: defaultdict(lambda: 0))
        vocab = self.vocabulary()

        # count all distinct tags associated to each word
        for sentence in self.corpus:
            for word, tag in sentence:
                if word in vocab:  # check if word is in vocabulary!
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
        Compute the initial state (first word tag) probabilities from a corpus of tagged
        sentences.

        Returns
        -------
        initial_state : defaultdict
            A vector representing the probabilities of the first word of a sentence being a
            certain type of POS tag, containing the log2 of each probability.
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
    def __init__(self, transition_matrix, emission_matrix, initial_state, tagset, vocabulary):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state
        self.tagset = tagset
        self.vocabulary = vocabulary

    def viterbi_best_path(self, sentence):
        """
        Compute the Viterbi algorithm over a sentence to find out its most probable POS tags.

        Parameters
        ----------
        sentence :
            A sequence of words to be tagged

        Returns
        -------
        viterbi : np.array
            Matrix with the probabilities computed at each step t
        best_path : list[tuple[str, str]]
            Sentence word by word with the most probable tags
        best_prob : float
            Probability of the best path found
        """
        # class vars initialization
        transitions = self.transition_matrix
        emissions = self.emission_matrix
        initial_state = self.initial_state
        tagset = self.tagset
        vocabulary = self.vocabulary

        # construct viterbi matrix
        words = [word if word in vocabulary else "UNK" for word in sentence.split(" ")]
        N, T = len(tagset), len(words)
        viterbi = np.full((N, T), -np.inf)
        backpointer = []

        # initialization step
        for idx, tag in enumerate(tagset):
            viterbi[idx, 0] = initial_state[tag] + emissions[words[0]][tag]
        best_arg = np.argmax(viterbi[:, 0])
        best_prob = viterbi[best_arg, 0]
        backpointer.append(best_arg)

        # compute next steps
        for idx_word, word in enumerate(words[1:], 1):  # note that it skips the first word
            for idx_tag, tag in enumerate(tagset):
                viterbi[idx_tag, idx_word] = (
                        best_prob + transitions[tagset[backpointer[-1]]][tag] + emissions[word][tag]
                )
            best_arg = np.argmax(viterbi[:, idx_word])
            best_prob = viterbi[best_arg, idx_word]
            backpointer.append(best_arg)

        # compute best path from backpointer
        tags = [tagset[i] for i in backpointer]
        best_path = list(zip(sentence.split(" "), tags))

        return viterbi, best_path, best_prob
