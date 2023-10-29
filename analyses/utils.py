import pandas as pd
import numpy as np
from collections import Counter

from IPython.display import display_html


def get_stats(dataset: list[list[tuple]]):
    sentence_lengths = [len(sentence) for sentence in dataset]

    print(f"Total sentences: {len(dataset)}")
    print(f"Average sentence length: {round(np.mean(sentence_lengths))}")
    print(f"Minimum sentence length: {min(sentence_lengths)}")
    print(f"Maximum sentence length: {max(sentence_lengths)}")
    print(f"Percentile 25, length: {np.percentile(sentence_lengths, 25)}")
    print(f"Percentile 50, length: {np.percentile(sentence_lengths, 50)}")
    print(f"Percentile 75, length: {np.percentile(sentence_lengths, 75)}")


def build_counts(dataset: list[list[tuple]]) -> tuple[Counter, Counter, Counter]:
    token_counts = Counter()
    tag_counts = Counter()
    pair_counts = Counter()

    for sentence in dataset:
        for token, tag in sentence:
            token_counts[token] += 1
            tag_counts[tag] += 1
            pair_counts[f"({token},{tag})"] += 1

    return token_counts, tag_counts, pair_counts


def build_dataframes(info: list[tuple]):
    tags = []
    tokens = []

    for sentence in info:
        for token, tag in sentence:
            tokens += [token]
            tags += [tag]
    return pd.DataFrame({"tokens": tokens, "tags": tags, "count": 1})


def print_top_tokens_given_tag(
    df: pd.DataFrame, tag: str, top: float = 5
) -> pd.DataFrame:
    return (
        df[df.tags == tag]
        .groupby(["tags", "tokens"])
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


def visualize_sample(dataset: list[list[tuple]], idx: int) -> dict:
    tokens = [token for token, tag in dataset[idx]]
    tags = [tag for token, tag in dataset[idx]]
    return pd.DataFrame({"tokens": tokens, "tags": tags}).T


def get_sentence_idx_given_pair(dataset: pd.DataFrame, pair: tuple[str, str]) -> int:
    for idx in range(len(dataset)):
        sentence = dataset[idx]
        if (pair) in sentence:
            return idx
