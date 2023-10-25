def parse_conllu_file(filepath):
    """
    Parse a CoNLL-U format file and extract words and their part-of-speech (POS) tags.

    Parameters
    ----------
    filepath : str
        The path to the CoNLL-U format file to be parsed.

    Returns
    -------
    list[list[tuple]]
        A list of sentences, where each sentence is represented as a list of
        (word, POS_tag) tuples.
    """
    # accumulators
    sentence, sentences = [], []

    # parsing
    with open(filepath, "r", encoding="utf-8") as conllu_file:
        for line in conllu_file:
            # transform to lowercase, delete whitespaces and skip comments
            line = line.lower().strip()
            if line.startswith("#"):
                continue

            # extract word and pos-tag creating a nested list of sentences
            columns = line.split("\t")
            if len(columns) != 1:
                sentence.append((columns[1], columns[3]))
            else:
                sentences.append(sentence)
                sentence = []

    return sentences
