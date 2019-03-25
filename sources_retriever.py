import numpy as np
from stft_tools import istft_vanilla
import scipy.io.wavfile as wav


def extract_sources_influences(my_s, my_a):
    n_freq_s, n_bins_s, n_sources_s = my_s.shape
    n_freq, n_canals, n_sources = my_a.shape

    assert n_freq == n_freq_s
    assert n_sources == n_sources_s

    sources_stft = np.zeros((n_sources, n_canals, n_freq, n_bins_s), dtype=np.complex128)
    for j in range(n_sources):
        for freq in range(n_freq):
            s_fj = my_s[freq, :, j].reshape((-1, 1))
            a_jf = my_a[freq, :, j].reshape((1, -1))
            # print(sources_stft.shape, s_fj.dot(a_jf).shape)
            # break
            sources_stft[j, :, freq, :] = s_fj.dot(a_jf).T
    return sources_stft


def to_files(sources_stft, filename, nperseg, fs):
    for j, source_j in enumerate(sources_stft):
        times, temporal = istft_vanilla(source_j, fs=fs, nperseg=nperseg)
        # break
        temporal = temporal.T
        temporal = temporal / np.abs(temporal).max()
        wav.write('{}_{}.wav'.format(filename, j), rate=fs, data=temporal)
