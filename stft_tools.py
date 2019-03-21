import numpy as np
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
import seaborn as sns

stft_vanilla = stft
istft_vanilla = istft


def stft_multi(x, **params):
    res = []
    for canal in x.T:
        freqs, times, x_stft = stft_vanilla(canal, **params)
        res.append(x_stft)
    return np.array(res)


def istft_multi(x_stft, **params):
    res = []
    n_canals = len(x_stft)
    for canal in x_stft:
        times, x_retrieved = istft_vanilla(canal, **params)
        res.append(x_retrieved)
    res = np.array(res).reshape((-1, n_canals))
    assert res.shape[1] == n_canals
    return res


def plot_spectogram(sig_power, frequencies=None, times=None):
    """

    :param sig_stft: Array of shape (fourier coefficients, segment times)
    :param frequencies:
    :param times:

    :return:
    """
    fig = plt.figure()
    ax = sns.heatmap(sig_power, robust=True)
    return fig, ax

