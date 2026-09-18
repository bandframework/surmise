import numpy as np
import matplotlib.figure as mfig

from estimate_sample_statistics import estimate_sample_statistics


class MplMcConvergenceVarND(mfig.Figure):
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

        self.suptitle(f"{dim}D Monte Carlo Sample Variance Convergence",
                      fontsize=self.fontsize_pt)

        for i in range(dim):
            subp = self.add_subplot(2, dim, i + 1)
            subp.set_title(rf"$X_{i+1}$", fontsize=self.fontsize_pt)
            subp.plot(steps, Cov_hat_n[:, i, i],
                      'k.', markersize=self.markersize_pt)
            subp.axhline(Cov[i, i], label=r"$\mathbb{V}$",
                         linestyle="--", linewidth=self.linewidth_pt,
                         color="red")
            subp.set_ylabel(r"$\hat{\sigma^2}_n$", fontsize=self.fontsize_pt)
            subp.legend(loc="best", fontsize=self.fontsize_pt)
            subp.grid(True)

            subp = self.add_subplot(2, dim, i + dim + 1)
            subp.semilogy(steps, np.fabs(1.0 - Cov_hat_n[:, i, i] / Cov[i, i]),
                          'k.', markersize=self.markersize_pt)
            subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
            subp.set_ylabel(r"$\hat{\sigma^2}_n$ Relative Error",
                            fontsize=self.fontsize_pt)
            subp.grid(True)

        self.tight_layout()
