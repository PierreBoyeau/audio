from audio_io import read_wav, save_signals
from stft_tools import stft_vanilla
from tqdm import tqdm
import matplotlib.pyplot as plt
from nmf_em_naive import NmfEmNaive, init_strategy
from sources_retriever import to_files, extract_sources_influences
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_components", type=int, default=12)
parser.add_argument("--n_sources", type=int, default=3)
parser.add_argument("--filesource")
parser.add_argument("--mode", default='D')
parser.add_argument("--directory", default="./results")
args = parser.parse_args()


n_components = args.n_components
n_sources = args.n_sources
filesource = args.filesource
direc = args.directory
mode = args.mode

fs, x_t = read_wav(filename=filesource)

print(x_t.shape)
# x_t = x_t[:40000]

freqs, times, x_f = stft_vanilla(x_t.T, nperseg=1024)
n_canals, n_freqs, n_bins = x_f.shape

# x_f = x_f.reshape((n_freqs, n_bins, n_canals))
x_f = x_f.transpose([1, 2, 0])
print(x_f.shape)

# plt.plot(x_t[:, 0])
# plt.show()

a0, w0, h0 = init_strategy(x_f, n_sources=n_sources, n_comps=n_components)

alg = NmfEmNaive(x_f, n_components, n_sources,
                 test_shapes=True, test_dots=False, mode=mode,
                 a0=a0, w0=w0, h0=h0, n_jobs=1)
print(alg.sigma_hat_f.min(), alg.sigma_hat_f.max())

costs = []

for iterate in tqdm(range(300)):
    alg.e_step()
    alg.m_step()
    my_cost = alg.cost()
    print(my_cost)
    costs.append(my_cost)

    signals_iter = alg.s
    a_iter = alg.a
    if iterate % 50 == 0:
        # save_signals(signals_iter, n_freqs=n_freqs, n_bins=n_bins, fs=fs,
        #              filename=direc+'/signals_iter{}'.format(iterate))
        coefs = extract_sources_influences(signals_iter, a_iter)
        to_files(coefs, n_freqs=n_freqs, n_bins=n_bins, fs=fs,
                     filename=direc+'/signals_iter{}'.format(iterate))


signals = alg.s
save_signals(signals, n_freqs=n_freqs, n_bins=n_bins, fs=fs,
             filename=direc+'/signals_final')

np.save(direc+'/final_s.npz', alg.s)
np.save(direc+'/final_a.npz', alg.a)
