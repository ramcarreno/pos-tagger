def parse_conllu_file(filepath):
    sentences = []
    sentence = []

    with open(filepath, 'r', encoding='utf-8') as conllu_file:
        for line in conllu_file:
            line = line.strip()
            if line.startswith("#"):
                continue

            columns = line.split("\t")

            if len(columns) != 1:
                sentence.append((columns[1], columns[3]))
            else:
                sentences.append(sentence)
                sentence = []

    return sentences

