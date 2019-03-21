import numpy as np
import scipy.io.wavfile as wav
import seaborn as sns

read_wav = wav.read


def write_wav(filename, rate, data):
    sig = np.round(data).astype(np.int16)
    wav.write(filename=filename, rate=rate, data=sig)


def float_to_bits_1d(arr, bits=16):
    N = len(arr)
    res = np.empty(shape=(N,), dtype=np.uint32)


