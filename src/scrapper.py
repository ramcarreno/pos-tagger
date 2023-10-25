def parse_conllu_file(filepath):
    """
    Parse a CoNLL-U format file and return list[list[tuple]] of word and pos-tag

    to be used as part of corpus preparation for the file pos_tagger.py

    """

    single_sentence = []
    all_sentences = []

    # lowcase, delete whitespaces and skips comments
    with open(filepath, "r", encoding="utf-8") as conllu_file:
        for line in conllu_file:
            line = line.lower().strip()
            if line.startswith("#"):
                continue

            # extract word and pos-tag creating a nested list of sentences
            columns = line.split("\t")
            if len(columns) != 1:
                single_sentence.append((columns[1], columns[3]))
            else:
                all_sentences.append(single_sentence)
                single_sentence = []

    return all_sentences
