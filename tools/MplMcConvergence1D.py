import numpy as np
import matplotlib.figure as mfig

from estimate_sample_statistics import estimate_sample_statistics


class MplMcConvergence(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fontsize_pt = 16
        self.markersize_pt = 5
        self.linewidth_pt = 1.5

    def draw_plot(self, randomized_samples, mu_true, var_true):
        n_samples = len(randomized_samples)
        steps = np.array([int(n) for n in np.linspace(0, n_samples, 101)])[1:]

        mean_checks = np.full(len(steps), np.nan, float)
        var_checks = mean_checks.copy()
        for i, n in enumerate(steps):
            mean_checks[i], var_checks[i] = \
                estimate_sample_statistics(randomized_samples[:n])

        self.clear()

        self.suptitle("Monte Carlo Convergence Check",
                      fontsize=self.fontsize_pt)

        subp = self.add_subplot(221)
        subp.plot(steps, mean_checks, 'k.', markersize=self.markersize_pt)
        subp.axhline(mu_true, label=r"$\mathbb{E}[X]$",
                     linestyle="--", linewidth=self.linewidth_pt, color="red")
        subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
        subp.set_ylabel(r"$\hat{\mu}_n$", fontsize=self.fontsize_pt)
        subp.legend(loc="best", fontsize=self.fontsize_pt)
        subp.grid(True)

        subp = self.add_subplot(222, sharex=subp)
        subp.plot(steps, var_checks, 'k.', markersize=self.markersize_pt)
        subp.axhline(var_true, label=r"$\mathbb{V}[X]$",
                     linestyle="--", linewidth=self.linewidth_pt, color="red")
        subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
        subp.set_ylabel(r"$\hat{\sigma^2}_n$", fontsize=self.fontsize_pt)
        subp.legend(loc="best", fontsize=self.fontsize_pt)
        subp.grid(True)

        subp = self.add_subplot(223, sharex=subp)
        subp.semilogy(steps, np.fabs(1.0 - mean_checks / mu_true),
                      'k.', markersize=self.markersize_pt)
        subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
        subp.set_ylabel(r"$\hat{\mu}_n$ Relative Error",
                        fontsize=self.fontsize_pt)
        subp.grid(True)

        subp = self.add_subplot(224, sharex=subp, sharey=subp)
        subp.semilogy(steps, np.fabs(1.0 - var_checks / var_true),
                      'k.', markersize=self.markersize_pt)
        subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
        subp.set_ylabel(r"$\hat{\sigma^2}_n$ Relative Error",
                        fontsize=self.fontsize_pt)
        subp.grid(True)

        self.tight_layout()
