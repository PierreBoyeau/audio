import numpy as np
from numpy.linalg import multi_dot
from joblib import Parallel, delayed


class NmfEmNaive:
    def __init__(self, x_f, n_components, n_sources,
                 a0=None, h0=None, w0=None,
                 dtype='complex128', test_shapes=False, test_dots=False,
                 mode='C', n_jobs=1, total_iter=300):
        """

        :param x_f:
        :param n_components: Number of signals in total
        We suppose that in each canal we will have k_j = k / nb_canals
        """
        self.n_jobs = n_jobs
        if n_jobs == 1:
            self.do_parallel = False
        elif n_jobs > 1:
            self.do_parallel = True
        else:
            raise ValueError

        self.dtype = dtype
        self.test_shapes = test_shapes
        self.test_dots = test_dots

        self.x_f = x_f
        self.x_f0 = x_f.copy()

        self.n_freq, self.n_bins, self.n_canals = self.x_f.shape

        self.n_comps = n_components
        self.n_sources = n_sources
        assert self.n_comps % self.n_sources == 0
        self.k_j = self.n_comps // self.n_sources
        self.k_to_j = np.repeat(np.arange(self.n_sources), self.k_j)
        assert self.k_to_j.shape == (self.n_comps,)

        # Useful quantities
        # Shape F, I, I

        self.r_xx0 = np.mean(self.x_f.reshape(self.n_freq, self.n_bins, self.n_canals, 1)
                            * self.x_f.conj().reshape(self.n_freq, self.n_bins, 1, self.n_canals),
                            axis=1)
        self.r_xx0 = 0.5 * (self.r_xx0 + self.r_xx0.transpose([0, 2, 1]))

        assert self.r_xx0.shape == (self.n_freq, self.n_canals, self.n_canals)
        self.r_xs = np.zeros(shape=(self.n_freq, self.n_canals, self.n_sources), dtype=self.dtype)
        self.r_ss = np.zeros(shape=(self.n_freq, self.n_sources, self.n_sources), dtype=self.dtype)
        self.u_k = np.zeros(shape=(self.n_freq, self.n_bins, self.n_comps), dtype=self.dtype)

        self._reshaped_x = self.x_f.reshape((self.n_freq, self.n_bins, 1, self.n_canals))

        # Parameters of the EM model
        # self.a = np.empty((self.n_canals, self.n_sources, self.n_freq))
        self.a = 2.0 ** (-0.5) * (np.random.randn(self.n_freq, self.n_canals, self.n_sources)
                                  + 1j * np.random.randn(self.n_freq, self.n_canals, self.n_sources)
                                  )
        self.a = self.a.astype(self.dtype) if a0 is None else a0
        if w0 is None:
            self.w = np.random.random((self.n_freq, self.n_comps)).astype(self.dtype)
        else:
            self.w = w0
        if h0 is None:
            self.h = np.random.random((self.n_comps, self.n_bins)).astype(self.dtype)
        else:
            self.h = h0
        self.normalize_awh()

        # Useful sigma for signal
        sigma_hat_f = np.abs(self.x_f) ** 2
        sigma_hat_f = sigma_hat_f.mean(axis=(1, 2))  # Mean over all axis except F, ie N and I
        sigma_hat_f = sigma_hat_f / 100.0
        self.sigma_hat_f = sigma_hat_f

        self.sigma_tilde = 3e-11

        self.mode = mode

        self.sigma_x_inv = None
        self.sigma_x = None

        self.s = None

        self.current_iter = 0
        self.total_iter = total_iter

    def e_step(self):
        """

        :return:
        """
        # if self.do_parallel:
        #     return self.e_step_parallel()
        x_f_iter = self.get_xf()

        # Part 1: sigma computations
        sigma_c = np.zeros((self.n_freq, self.n_bins, self.n_comps), dtype=self.dtype)
        sigma_x = np.zeros((self.n_freq, self.n_bins, self.n_canals, self.n_canals),
                           dtype=self.dtype)
        sigma_s = np.zeros((self.n_freq, self.n_bins, self.n_sources), dtype=self.dtype)
        my_sigma_b = self.get_sigma_b()

        for freq in np.arange(self.n_freq):
            for n in np.arange(self.n_bins):
                sigma_c[freq, n, :] = self.w[freq, :] * self.h[:, n]

        for j in range(self.n_sources):
            k_indices = np.where(self.k_to_j == j)[0]
            sigma_s[:, :, j] = sigma_c[:, :, k_indices].sum(axis=-1)

        for freq in np.arange(self.n_freq):
            for n in np.arange(self.n_bins):
                a_f = self.a[freq]
                sigma_x[freq, n, :] = (multi_dot([a_f, np.diag(sigma_s[freq, n, :]), a_f.conj().T])
                                       + my_sigma_b[freq])

        # sigma_x = sigma_x.real.astype(self.dtype)
        sigma_x_inv = np.linalg.inv(sigma_x)
        # sigma_x_inv = sigma_x_inv.real.astype(self.dtype)

        self.sigma_x = sigma_x
        self.sigma_x_inv = sigma_x_inv

        # G computations
        g_s = np.zeros((self.n_freq, self.n_bins, self.n_sources, self.n_canals), dtype=self.dtype)
        g_c = np.zeros((self.n_freq, self.n_bins, self.n_comps, self.n_canals), dtype=self.dtype)

        a_round = self.get_around()

        for freq in np.arange(self.n_freq):
            for n in np.arange(self.n_bins):
                a_f_h = self.a[freq, :].conj().T
                a_round_f_h = a_round[freq, :].conj().T

                sigma_s_fn_diag = np.diag(sigma_s[freq, n])
                sigma_c_fn_diag = np.diag(sigma_c[freq, n])

                g_s_fn = multi_dot([sigma_s_fn_diag, a_f_h, sigma_x_inv[freq, n]])
                g_c_fn = multi_dot([sigma_c_fn_diag, a_round_f_h, sigma_x_inv[freq, n]])

                assert g_s_fn.shape == (self.n_sources, self.n_canals)
                assert g_c_fn.shape == (self.n_comps, self.n_canals)

                g_s[freq, n, :] = g_s_fn
                g_c[freq, n, :] = g_c_fn

        # S and C computation
        s = np.zeros((self.n_freq, self.n_bins, self.n_sources), dtype=self.dtype)
        c = np.zeros((self.n_freq, self.n_bins, self.n_comps), dtype=self.dtype)
        for freq in np.arange(self.n_freq):
            for n in np.arange(self.n_bins):
                x_fn = x_f_iter[freq, n]
                g_s_fn = g_s[freq, n]
                g_c_fn = g_c[freq, n]

                s_fn = g_s_fn.dot(x_fn)
                c_fn = g_c_fn.dot(x_fn)

                assert s_fn.shape == (self.n_sources,)
                assert c_fn.shape == (self.n_comps,)

                s[freq, n, :] = s_fn
                c[freq, n, :] = c_fn

        self.s = s

        # Cov computations
        s_conj_tensor = s.conj().reshape((self.n_freq, self.n_bins, 1, self.n_sources))
        self.r_xs = np.mean(x_f_iter.reshape((self.n_freq, self.n_bins, self.n_canals, 1))
                            * s_conj_tensor,
                            axis=1)

        ss_all = np.zeros(shape=(self.n_freq, self.n_bins, self.n_sources, self.n_sources),
                          dtype=self.dtype)
        for freq in np.arange(self.n_freq):
            for n in np.arange(self.n_bins):
                s_fn = s[freq, n, :].reshape((-1, 1))
                sigma_s_fn_diag = np.diag(sigma_s[freq, n])

                last_term_ss = multi_dot([g_s[freq, n, :],
                                          self.a[freq, :],
                                          sigma_s_fn_diag])
                sst = s_fn.dot(s_fn.conj().T)

                assert last_term_ss.shape == (self.n_sources, self.n_sources)
                assert sst.shape == (self.n_sources, self.n_sources)

                ss_all[freq, n, :] = (sst + np.diag(sigma_s[freq, n, :])
                                      - last_term_ss)

                af_sigma_c_prod = a_round[freq, :].dot(np.diag(sigma_c[freq, n]))
                last_term = (g_c[freq, n, :] * af_sigma_c_prod.T).sum(axis=-1)
                self.u_k[freq, n, :] = (c[freq, n, :] * c[freq, n, :].conj() + sigma_c[freq, n]
                                        - last_term)
        self.u_k = self.u_k.real.astype(self.dtype)

        self.r_ss = ss_all.mean(axis=1)
        self.r_ss = 0.5 * (self.r_ss + self.r_ss.transpose([0, 2, 1]))

        if self.test_shapes:
            assert sigma_c.shape == (self.n_freq, self.n_bins, self.n_comps)
            assert sigma_s.shape == (self.n_freq, self.n_bins, self.n_sources)
            assert sigma_x.shape == (self.n_freq, self.n_bins, self.n_canals, self.n_canals)
            assert sigma_x_inv.shape == (self.n_freq, self.n_bins, self.n_canals, self.n_canals)
            assert a_round.shape == (self.n_freq, self.n_canals, self.n_comps), a_round.shape
            assert g_s.shape == (self.n_freq, self.n_bins, self.n_sources, self.n_canals)
            assert g_c.shape == (self.n_freq, self.n_bins, self.n_comps, self.n_canals)
            assert s.shape == (self.n_freq, self.n_bins, self.n_sources)
            assert c.shape == (self.n_freq, self.n_bins, self.n_comps)
            assert self.r_xs.shape == (self.n_freq, self.n_canals, self.n_sources), self.r_xs.shape
            assert self.r_ss.shape == (self.n_freq, self.n_sources, self.n_sources)
            assert self.u_k.shape == (self.n_freq, self.n_bins, self.n_comps)

        return

    def m_step(self):
        """
        Maximization step
        :return:
        """
        # print(np.linalg.norm(self.a))

        # A computation
        r_ss_inv = np.linalg.inv(self.r_ss)
        for freq in np.arange(self.n_freq):
            r_ss_f_inv = r_ss_inv[freq]
            self.a[freq, :] = self.r_xs[freq, :].dot(r_ss_f_inv)

        # W computation
        h_T = self.h.T
        h_T = np.expand_dims(h_T, 0)
        new_w = self.u_k / h_T
        new_w = new_w.real
        new_w = new_w.mean(axis=1)

        # H computation
        new_h = self.u_k / np.expand_dims(new_w, 1)
        new_h = new_h.real
        # new_h = new_h.mean(axis=0).reshape((self.n_comps, self.n_bins))
        new_h = new_h.mean(axis=0).T

        # self.sigma_b = new_sigma
        self.w = new_w
        self.h = new_h
        self.normalize_awh()

        if self.test_shapes:
            assert self.a.shape == (self.n_freq, self.n_canals, self.n_sources)
            # assert self.sigma_b.shape == (self.n_freq, self.n_canals, self.n_canals)
            assert self.w.shape == (self.n_freq, self.n_comps)
            assert self.h.shape == (self.n_comps, self.n_bins)

            print('a norm: ', np.linalg.norm(self.a))
            print('h norm: ', np.linalg.norm(self.h))
            print('w norm: ', np.linalg.norm(self.w))

        self.current_iter += 1

        return


    def normalize_awh(self):
        """
        Normalizes a, w, h based on paper [1] recommandations
        w FK
        h KN
        :return:

        """
        # First Step: a, h normalization
        scale = np.abs(self.a)
        scale = np.sum(scale ** 2, axis=1)
        new_scale = np.sqrt(scale)

        # TODO: Verify all this
        a_sign = np.sign(self.a[:, 0, :])
        new_a = self.a * np.expand_dims(a_sign, 1)
        new_a = new_a / new_scale.reshape((self.n_freq, 1, self.n_sources))
        new_w = np.empty(shape=(self.n_freq, self.n_comps), dtype=self.dtype)
        for j in range(self.n_sources):
            k_indices = np.where(self.k_to_j == j)[0]
            new_w[:, k_indices] = self.w[:, k_indices] * new_scale[:, j].reshape((self.n_freq, 1))

        self.a = new_a
        self.w = new_w

        # Second Step: w, h normalization
        norm_term = self.w.sum(axis=0)  #  Sum over all frequencies ==> (n_comps,)
        new_w = self.w.dot(np.diag(1.0/norm_term))
        # print(self.h.shape, self.n_comps, self.n_bins)
        new_h = np.diag(norm_term).dot(self.h)
        assert new_h.shape == (self.n_comps, self.n_bins)

        self.w = new_w.real
        self.h = new_h.real
        pass

    def cost(self):
        my_cost = 0
        for freq in np.arange(self.n_freq):
            for n in np.arange(self.n_bins):
                sigma_x_fn_inv = self.sigma_x_inv[freq, n, :]
                sigma_x_fn = self.sigma_x[freq, n, :]

                x_fn = self.x_f[freq, n, :].reshape((-1, 1))
                my_cost += (self.x_f[freq, n, :].conj().T.dot(sigma_x_fn_inv.dot(x_fn))).real
                my_cost += np.log(np.linalg.det(sigma_x_fn)).real
        return my_cost

    def get_around(self):
        """

        :return: A_round matrix
        """
        a_round = np.empty((self.n_freq, self.n_canals, self.n_comps), dtype=self.dtype)
        for k, j in enumerate(self.k_to_j):
            a_round[:, :, k] = self.a[:, :, j]
        return a_round

    def get_sigma_b(self):
        """

        :return: (n_freq, n_canals, n_canals) tensor
        """
        if self.mode == 'B':
            cov = self.sigma_hat_f.reshape((-1, 1, 1))
            cov = cov * np.ones(shape=(self.n_freq, self.n_canals, self.n_canals))
            return cov.astype(self.dtype)

        elif self.mode == 'C' or self.mode == 'D':
            cov = self.sigma_hat_f.reshape((-1, 1, 1))
            cov = ((np.sqrt(cov) * (self.total_iter - self.current_iter)
                    + np.sqrt(self.sigma_tilde) * self.current_iter)
                   / self.total_iter)
            cov = cov ** 2
            cov = cov * np.ones(shape=(self.n_freq, self.n_canals, self.n_canals))
            return cov.astype(self.dtype)
        else:
            raise KeyError

    def get_xf(self):
        if self.mode == 'D':
            # FII
            sigma_b = self.get_sigma_b()
            new_xf = np.zeros(shape=(self.n_freq, self.n_bins, self.n_canals),
                              dtype=self.dtype)
            zeros = np.zeros(self.n_canals)
            for freq in np.arange(self.n_freq):
                noise_f = np.random.multivariate_normal(mean=zeros, cov=sigma_b[freq, :],
                                                        size=self.n_bins)
                # print(noise_f.shape,self.x_f0[freq, :].shape)
                new_xf[freq, :] = (self.x_f0[freq, :] + noise_f)
            return new_xf

        else:
            return self.x_f0

    @staticmethod
    def mat_norm(a_mat, b_mat):
        assert a_mat.shape == b_mat.shape, (a_mat.shape, b_mat.shape)
        return np.abs(a_mat - b_mat).max()


def init_strategy(my_x, n_sources, n_comps):
    n_freq, n_bins, n_canals = my_x.shape
    my_std = 0.5 * np.mean(np.abs(my_x) ** 2, axis=(1, 2))  # FNI

    a_0 = 0.5 * (1.9 * np.abs(np.random.randn(1, n_canals, n_sources))
                 + 0.1 * np.ones((n_freq, n_canals, n_sources))) * np.sign(
        np.random.randn(1, n_canals, n_sources)) + 1j * np.random.randn(1, n_canals, n_sources)

    w_0 = 0.5 * (np.abs(np.random.randn(n_freq, n_comps)) + 1.0) * my_std.reshape((-1, 1))
    h_0 = 0.5 * (np.abs(np.random.randn(n_comps, n_bins)) + 1.0)

    return a_0, w_0, h_0


# if __name__ == '__main__':
#     from audio_io import read_wav, save_signals
#     from stft_tools import stft_vanilla
#     from tqdm import tqdm
#     import matplotlib.pyplot as plt
#
#     n_components = 12
#     n_sources = 3
#
#     fs, x_t = read_wav(filename='./data/dev2/dev2_female4_inst_mix.wav')
#
#     print(x_t.shape)
#     # x_t = x_t[:40000]
#     freqs, times, x_f = stft_vanilla(x_t.T, nperseg=1024)
#     n_canals, n_freqs, n_bins = x_f.shape
#
#     # x_f = x_f.reshape((n_freqs, n_bins, n_canals))
#     x_f = x_f.transpose([1, 2, 0])
#     print(x_f.shape)
#
#     # plt.plot(x_t[:, 0])
#     # plt.show()
#
#     a0, w0, h0 = init_strategy(x_f, n_sources=n_sources, n_comps=n_components)
#
#     alg = NmfEmNaive(x_f, n_components, n_sources, test_shapes=True, test_dots=False, mode='D',
#                      a0=a0, w0=w0, h0=h0, n_jobs=1)
#     print(alg.sigma_hat_f.min(), alg.sigma_hat_f.max())
#
#     costs = []
#
#     for iterate in tqdm(range(300)):
#         alg.e_step()
#         alg.m_step()
#         my_cost = alg.cost()
#         print(my_cost)
#         costs.append(my_cost)
#
#         signals_iter = alg.s
#         if iterate % 50 == 0:
#             save_signals(signals_iter, n_freqs=n_freqs, n_bins=n_bins, fs=fs,
#                          filename='results/D/signals_iter{}'.format(iterate))
#
#     signals = alg.s
#     save_signals(signals, n_freqs=n_freqs, n_bins=n_bins, fs=fs,
#                  filename='results/D/signals_final')
