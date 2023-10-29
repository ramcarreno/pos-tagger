from collections import Counter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def plot_frequency_of_(
    feature: str, feature_counts_train: Counter, feature_counts_test: Counter, top=50
):
    most_common_counts_train = feature_counts_train.most_common(top)
    x_train = [word for word, _ in most_common_counts_train]
    y_train = [count for _, count in most_common_counts_train]

    most_common_counts_test = feature_counts_test.most_common(top)
    x_test = [word for word, _ in most_common_counts_test]
    y_test = [count for _, count in most_common_counts_test]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Train data", "Test data"])
    fig.add_trace(go.Bar(x=x_train, y=y_train), row=1, col=1)
    fig.add_trace(go.Bar(x=x_test, y=y_test), row=1, col=2)

    fig.update_xaxes(title_text=f"{feature.title()}", row=1, col=1)
    fig.update_yaxes(title_text="Ocurrences", row=1, col=1)
    fig.update_xaxes(title_text=f"{feature.title()}", row=1, col=2)
    fig.update_yaxes(title_text="Ocurrences", row=1, col=2)

    fig.update_layout(showlegend=False)
    fig.show()
