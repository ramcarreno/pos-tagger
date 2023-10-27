import pandas as pd
import numpy as np
from collections import Counter

from IPython.display import display_html


def get_stats(dataset: list[list[tuple]]):
    sentence_lenghts = [len(sentence) for sentence in dataset]

    print(f"Total sentences: {len(dataset)}")
    print(f"Average sentence length: {round(np.mean(sentence_lenghts))}")
    print(f"Minimum sentence length: {min(sentence_lenghts)}")
    print(f"Maximum sentence length: {max(sentence_lenghts)}")
    print(f"Percentile 25, lenght: {np.percentile(sentence_lenghts, 25)}")
    print(f"Percentile 50, lenght: {np.percentile(sentence_lenghts, 50)}")
    print(f"Percentile 75, lenght: {np.percentile(sentence_lenghts, 75)}")


def build_counts(dataset: list[list[tuple]]) -> tuple[Counter, Counter, Counter]:
    word_counts = Counter()
    tag_counts = Counter()
    pair_counts = Counter()

    for sentence in dataset:
        for word, tag in sentence:
            word_counts[word] += 1
            tag_counts[tag] += 1
            pair_counts[f"({word},{tag})"] += 1

    return word_counts, tag_counts, pair_counts


def build_dataframes(info: list[tuple]):
    tags = []
    words = []

    for sentence in info:
        for word, tag in sentence:
            words += [word]
            tags += [tag]
    return pd.DataFrame({"words": words, "tags": tags, "count": 1})


def print_top_words_given_tag(
    df: pd.DataFrame, tag: str, top: float = 5
) -> pd.DataFrame:
    return (
        df[df.tags == tag]
        .groupby(["tags", "words"])
        .sum()
        .reset_index()
        .sort_values(by=["tags", "count"], ascending=False)[:top]
        .reset_index(drop=True)
    )


def display_side_by_side(*args):
    html_str = ""
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace("table", 'table style="display:inline"'), raw=True)
