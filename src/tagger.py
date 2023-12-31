from collections import defaultdict
from functools import reduce

import numpy as np
from typing import List


class HiddenMarkovModel:
    def __init__(self, corpus: List[List[tuple]]):
        """
        Initialize a Hidden Markov Model with a given corpus.

        Parameters
        ----------
        corpus : List[List[tuple]]
            A list of sentences, where each sentence is represented as
            a list of (word, POS tag) tuples.

        Notes
        -----
        This constructor initializes the Hidden Markov Model with the provided corpus,
        which will be used for training the HMM.
        """
        self.corpus = corpus

    def train(self):
        """
        Train the Hidden Markov Model based on the provided corpus and
        return a HiddenMarkovModelTagger.

        Returns
        -------
        hmm_tagger : HiddenMarkovModelTagger
            A HiddenMarkovModelTagger trained with the transition matrix, emission matrix,
            initial state, POS tagset, and vocabulary derived from the corpus.

        Notes
        -----
        This method trains the Hidden Markov Model by computing the transition matrix,
        emission matrix, initial state probabilities, POS tagset, and vocabulary from the
        given corpus. It then creates and returns a HiddenMarkovModelTagger instance with
        these parameters for later use in POS tagging tasks.
        """
        return HiddenMarkovModelTagger(
            self.transition_matrix(),
            self.emission_matrix(),
            self.initial_state(),
            self.tagset(),
            self.vocabulary(),
        )

    def vocabulary(self, unk_threshold=3):
        """
        Create a vocabulary of words based on word frequencies in the corpus.

        Parameters
        ----------
        unk_threshold : int, optional
            The frequency threshold below which words are considered rare and replaced
            with the 'UNK' token. Words appearing less than or equal to `unk_threshold`
            times are replaced with 'UNK'. Default value is 3 appearances.

        Returns
        -------
        vocab : set
            A set containing the vocabulary of words derived from the corpus.
            The vocabulary includes common words and a special 'UNK' token
            for rare words (below the threshold).
        """
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
        """
        Get the set of all unique POS tags found in the corpus.

        Returns
        -------
        tagset : tuple
            A sorted tuple containing all distinct POS tags present in the corpus.
            The tags are sorted in alphabetical order to maintain a consistent order.
        """
        # init
        tagset = set()

        # save all possible tags found in corpus
        for sentence in self.corpus:
            for word, tag in sentence:
                tagset.add(tag)

        # convert resulting set to tuple so an order is kept
        tagset = tuple(sorted(tagset))
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
        transition_matrix = defaultdict(
            lambda: defaultdict(lambda: -np.inf)
        )  # probability 0 to logprob -> -inf
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
    def __init__(
        self, transition_matrix, emission_matrix, initial_state, tagset, vocabulary
    ):
        """
        Initialize a Hidden Markov Model Tagger with its essential parameters.

        Parameters
        ----------
        transition_matrix : defaultdict[defaultdict]
            A matrix representing the probabilities of transitioning from one POS tag to another,
            containing the log2 of each probability.

        emission_matrix : defaultdict[defaultdict]
            A matrix representing the probabilities of emitting words from POS tags,
            containing the log2 of each probability.

        initial_state : defaultdict
            A dictionary representing the initial state probabilities for each POS tag.

        tagset : tuple
            A sorted tuple containing all distinct POS tags present in the corpus.

        vocabulary : set
            A set containing the vocabulary of words derived from the corpus.

        Notes
        -----
        This constructor initializes a Hidden Markov Model Tagger with the key parameters necessary for
        part-of-speech tagging. The `transition_matrix` represents the transition probabilities between
        POS tags, the `emission_matrix` represents the probabilities of emitting words from POS tags,
        the `initial_state` represents the initial state probabilities for each POS tag, the `tagset` is a
        tuple of distinct POS tags, and the `vocabulary` is a set of words derived from the corpus.
        """
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
        viterbi_matrix : np.array
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
        viterbi_matrix = np.full((N, T), -np.inf)
        backpointer = []

        # initialization step
        for idx, tag in enumerate(tagset):
            viterbi_matrix[idx, 0] = initial_state[tag] + emissions[words[0]][tag]
        best_arg = np.argmax(viterbi_matrix[:, 0])
        best_prob = viterbi_matrix[best_arg, 0]
        backpointer.append(best_arg)

        # compute next steps
        for idx_word, word in enumerate(
            words[1:], 1
        ):  # note that it skips the first word
            for idx_tag, tag in enumerate(tagset):
                viterbi_matrix[idx_tag, idx_word] = (
                    best_prob
                    + transitions[tagset[backpointer[-1]]][tag]
                    + emissions[word][tag]
                )
            best_arg = np.argmax(viterbi_matrix[:, idx_word])
            best_prob = viterbi_matrix[best_arg, idx_word]
            backpointer.append(best_arg)

        # compute best path from backpointer
        tags = [tagset[i] for i in backpointer]
        best_path = list(zip(sentence.split(" "), tags))

        return viterbi_matrix, best_path, best_prob

    def predict(self, corpus):
        """
        Perform POS tagging on a given corpus of sentences.

        Parameters
        ----------
        corpus : List[List[tuple]]
            A list of sentences, where each sentence is represented as a list of (word, POS tag) tuples.

        Returns
        -------
        corpus_prediction : List[List[tuple]]
            A list of tagged sentences, where each sentence is represented as a list of
            (word, predicted POS tag) tuples.
        """
        corpus_prediction = []
        for sentence in corpus:
            s = reduce(lambda x, y: x + " " + y, map(lambda x: x[0], sentence))
            _, s_p, _ = self.viterbi_best_path(s)
            corpus_prediction.append(s_p)

        return corpus_prediction

    def get_confusion_matrix(self, corpus, corpus_prediction):
        """
        Calculate the confusion matrix for POS tagging accuracy assessment.

        Parameters
        ----------
        corpus : List[List[tuple]]
            A list of sentences, where each sentence is represented as a list of (word, POS tag) tuples.
            This represents the expected POS tag annotations.

        corpus_prediction : List[List[tuple]]
            A list of tagged sentences, where each sentence is represented as a list of
            (word, predicted POS tag) tuples.

        Returns
        -------
        confusion_matrix : numpy.ndarray
            A 2D numpy array representing the confusion matrix, which provides a count of expected POS tags
            versus predicted POS tags.
        """
        tagset = self.tagset

        N = len(tagset)
        confusion_matrix = np.zeros((N, N)).astype(int)
        for i in range(len(corpus)):
            expected, prediction = corpus[i], corpus_prediction[i]
            for token, token_predicted in zip(
                map(lambda x: x[1], expected), map(lambda x: x[1], prediction)
            ):
                confusion_matrix[
                    tagset.index(token), tagset.index(token_predicted)
                ] += 1

        return confusion_matrix
