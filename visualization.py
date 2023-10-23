import numpy as np
import plotly.graph_objects as go


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
