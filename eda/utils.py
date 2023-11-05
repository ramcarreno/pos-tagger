import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from PIL import Image

from IPython.display import display_html


# Print basic information for data exploration
def get_stats(dataset: list[list[tuple]]):
    sentence_lengths = [len(sentence) for sentence in dataset]

    print(f"Total sentences: {len(dataset)}")
    print(f"Average sentence length: {round(np.mean(sentence_lengths))}")
    print(f"Minimum sentence length: {min(sentence_lengths)}")
    print(f"Maximum sentence length: {max(sentence_lengths)}")
    print(f"Percentile 25, length: {np.percentile(sentence_lengths, 25)}")
    print(f"Percentile 50, length: {np.percentile(sentence_lengths, 50)}")
    print(f"Percentile 75, length: {np.percentile(sentence_lengths, 75)}")


# Build token, tag, and token-tag pair counts excluding stopwords and common punctuation for visualization
def build_counts_nostop(
    dataset: list[list[tuple]], language: str
) -> tuple[Counter, Counter, Counter]:
    stop_words = set(stopwords.words(language))
    punctuation = [
        ",",
        ".",
        "(",
        ")",
        "-",
        "?",
        "¿",
        "!",
        "¡",
        "'s",
        ":",
        "[",
        "]",
        '"',
        "'",
        "–",
        "—",
        "'s",
        "”",
        "“",
        "l'",
        ";",
        "s'",
        "d'",
        "_",
    ]
    token_counts = Counter()
    tag_counts = Counter()
    pair_counts = Counter()

    for sentence in dataset:
        for token, tag in sentence:
            if token not in stop_words and token not in punctuation:
                token_counts[token] += 1
                tag_counts[tag] += 1
                pair_counts[f"({token},{tag})"] += 1

    return token_counts, tag_counts, pair_counts


# Build token, tag, and token-tag pair counts withouth excluding stopwords or punctuation for plotting
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


# Create dictionaries for a better visualization
def build_dataframes(info: list[tuple]):
    tags = []
    tokens = []

    for sentence in info:
        for token, tag in sentence:
            tokens += [token]
            tags += [tag]
    return pd.DataFrame({"tokens": tokens, "tags": tags, "count": 1})


# Print the five most common tokens given a tag
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


# Display several tables next side by side
def display_side_by_side(*args):
    html_str = ""
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace("table", 'table style="display:inline"'), raw=True)


# Create a dictionary for a better visualization
def visualize_sample(dataset: list[list[tuple]], idx: int) -> dict:
    tokens = [token for token, tag in dataset[idx]]
    tags = [tag for token, tag in dataset[idx]]
    return pd.DataFrame({"tokens": tokens, "tags": tags}).T


# Return sentence with its tags for its manual evaluation
def get_sentence_idx_given_pair(dataset: pd.DataFrame, pair: tuple[str, str]) -> int:
    for idx in range(len(dataset)):
        sentence = dataset[idx]
        if (pair) in sentence:
            return idx


# Generate a wordcloud imagen
def wordcloud(train_info: list[list[tuple]], language: str):
    # Remove stopwords
    stop_words = set(stopwords.words(language))
    if language == "catalan":
        tokens_word_cloud = [
            w[0][0]
            for w in train_info
            if not w[0][0] in stop_words and len(w[0][0]) > 2
        ]
    else:
        tokens_word_cloud = [w[0][0] for w in train_info if not w[0][0] in stop_words]

    # Load an image as a mask for the word cloud
    mask = np.array(Image.open("mask.png"))
    mask[mask == 1] = 255

    # Lowercase and create a single string with all tokens for WordCloud generation
    comment_words = ""
    for i in range(len(tokens_word_cloud)):
        tokens_word_cloud[i] = tokens_word_cloud[i].lower()
        comment_words += " ".join(tokens_word_cloud) + " "

    # Specify settings for the word cloud generation
    wordcloud = WordCloud(
        background_color="white",
        min_font_size=9,
        max_font_size=150,
        max_words=60,
        collocations=False,
        mask=mask,
    ).generate(comment_words)

    # Plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()
