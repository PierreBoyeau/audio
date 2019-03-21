import numpy as np
import scipy.io.wavfile as wav
import seaborn as sns

from stft_tools import istft_vanilla

read_wav = wav.read


def write_wav(filename, rate, data):
    sig = np.round(data).astype(np.int16)
    wav.write(filename=filename, rate=rate, data=sig)


def save_signals(signals, n_freqs, n_bins, fs, filename):
    signals = signals.transpose([2, 0, 1])
    for j, signal in enumerate(signals):
        assert signal.shape == (n_freqs, n_bins)
        times, temporal = istft_vanilla(signal, fs=fs)
        temporal = temporal / np.abs(temporal).max()
        wav.write('{}_{}.wav'.format(filename, j), rate=fs, data=temporal)

