import numpy as np
import matplotlib.figure as mfig

from estimate_sample_statistics import estimate_sample_statistics


class MplMcConvergenceMeansND(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fontsize_pt = 16
        self.markersize_pt = 5
        self.linewidth_pt = 1.5

    def draw_plot(self, randomized_samples, mean):
        n_samples = len(randomized_samples)

        assert mean.ndim == 1
        dim = len(mean)

        steps = np.array([int(n) for n in np.linspace(0, n_samples, 101)])[1:]
        n_steps = len(steps)

        mean_hat_n = np.full([n_steps, dim], np.nan, float)
        for i, n in enumerate(steps):
            mean_hat_n[i, :], _ = \
                estimate_sample_statistics(randomized_samples[:n])

        self.clear()

        self.suptitle(f"{dim}D Monte Carlo Sample Mean Convergence",
                      fontsize=self.fontsize_pt)

        for i in range(dim):
            subp = self.add_subplot(2, dim, i + 1)
            subp.set_title(rf"$X_{i+1}$", fontsize=self.fontsize_pt)
            subp.plot(steps, mean_hat_n[:, i],
                      'k.', markersize=self.markersize_pt)
            subp.axhline(mean[i], label=r"$\mathbb{E}$",
                         linestyle="--", linewidth=self.linewidth_pt,
                         color="red")
            subp.set_ylabel(r"$\hat{\mu}_n$", fontsize=self.fontsize_pt)
            subp.legend(loc="best", fontsize=self.fontsize_pt)
            subp.grid(True)

            subp = self.add_subplot(2, dim, i + dim + 1)
            subp.semilogy(steps, np.fabs(1.0 - mean_hat_n[:, i] / mean[i]),
                          'k.', markersize=self.markersize_pt)
            subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
            subp.set_ylabel(r"$\hat{\mu}_n$ Relative Error",
                            fontsize=self.fontsize_pt)
            subp.grid(True)

        self.tight_layout()
