from collections import Counter
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_viterbi_path_binary(
    viterbi: np.array, backpointer: list, sentence: str, tags: str
) -> None:
    """
    Shows the viterbi matrix result, highlighting the values that obtained the maximum probability at a given
    time step.

    Parameters
    ----------
    viterbi: viterbi algorithm matrix result. First element obtained after calling fuction: viterbi_logprobs
    backpointer: viterbi algorithm backpointer result. Second element obtained after calling fuction: viterbi_logprobs
    sentence: sentence which we have computed the probabilities against
    tags: set of tags used in the algorithm

    """
    colors = viterbi.copy()

    for i in range(len(backpointer)):
        argmax_row = backpointer[i]
        colors[argmax_row, i] += 10000

    fig = go.Figure(
        data=go.Heatmap(
            z=colors,
            x=[word for word in sentence.split(" ")],
            y=[tag for tag in tags],
            colorscale="Thermal",
            text=viterbi,
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=False,
        )
    )
    fig.show(legend=False)


def plot_viterbi_matrix(viterbi: np.array, sentence: str, tags: str) -> None:
    """
    Shows the viterbi matrix result, painting the probabilities overall

    Parameters
    ----------
    viterbi: viterbi algorithm matrix result. First element obtained after calling fuction: viterbi_logprobs
    sentence: sentence which we have computed the probabilities against
    tags: set of tags used in the algorithm

    """
    fig = go.Figure(
        data=go.Heatmap(
            z=viterbi,
            x=[word for word in sentence.split(" ")],
            y=[tag for tag in tags],
            colorscale="Thermal",
        )
    )
    fig.show(legend=False)


def plot_frequency_of_(feature: str, feature_counts: Counter, top=50):
    most_common_counts = feature_counts.most_common(top)
    x = [word for word, _ in most_common_counts]
    y = [count for _, count in most_common_counts]

    fig = px.bar(x=x, y=y, title=f"Top {top} {feature} of the dataset")
    fig.update_xaxes(title=f"{feature}".title())
    fig.update_yaxes(title=f"Occurrences")
    fig.show()
