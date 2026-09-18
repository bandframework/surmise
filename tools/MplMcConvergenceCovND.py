import numpy as np
import itertools as it
import matplotlib.figure as mfig

from estimate_sample_statistics import estimate_sample_statistics


class MplMcConvergenceCovND(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fontsize_pt = 16
        self.markersize_pt = 5
        self.linewidth_pt = 1.5

    def draw_plot(self, randomized_samples, Cov):
        n_samples = len(randomized_samples)

        assert Cov.ndim == 2
        assert Cov.shape[1] == Cov.shape[0]
        dim = Cov.shape[0]
        assert np.array_equal(Cov, Cov.T)
        assert all(np.diag(Cov) > 0.0)

        steps = np.array([int(n) for n in np.linspace(0, n_samples, 101)])[1:]
        n_steps = len(steps)

        Cov_hat_n = np.full([n_steps, dim, dim], np.nan, float)
        for i, n in enumerate(steps):
            _,  Cov_hat_n[i, :, :] = \
                estimate_sample_statistics(randomized_samples[:n])

        self.clear()

        self.suptitle(f"{dim}D Monte Carlo Sample Covariance Convergence",
                      fontsize=self.fontsize_pt)

        n_pairs = int(0.5 * (dim - 1) * dim)
        for k, (i, j) in enumerate(it.combinations(range(dim), 2)):
            assert j > i
            if Cov[i, j] <= 1.0e-3:
                cov_err = np.fabs(Cov_hat_n[:, i, j] - Cov[i, j])
                cov_err_label = "Absolute Error"
            else:
                cov_err = np.fabs(1.0 - Cov_hat_n[:, i, j] / Cov[i, j])
                cov_err_label = "Relative Error"

            subp = self.add_subplot(2, n_pairs, k + 1)
            subp.set_title(rf"$X_{i+1} X_{j+1}$", fontsize=self.fontsize_pt)
            subp.plot(steps, Cov_hat_n[:, i, j],
                      'k.', markersize=self.markersize_pt)
            subp.axhline(Cov[i, j], label=r"$Cov$",
                         linestyle="--", linewidth=self.linewidth_pt,
                         color="red")
            subp.set_ylabel(r"$\hat{Cov}_n$", fontsize=self.fontsize_pt)
            subp.legend(loc="best", fontsize=self.fontsize_pt)
            subp.grid(True)

            subp = self.add_subplot(2, n_pairs, k + n_pairs + 1)
            subp.semilogy(steps, cov_err,
                          'k.', markersize=self.markersize_pt)
            subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
            subp.set_ylabel(rf"$\hat{{Cov}}_n$ {cov_err_label}",
                            fontsize=self.fontsize_pt)
            subp.grid(True)

        self.tight_layout()
