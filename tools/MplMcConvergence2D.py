import numpy as np
import matplotlib.figure as mfig

from estimate_sample_statistics import estimate_sample_statistics


class MplMcConvergence2D(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fontsize_pt = 16
        self.markersize_pt = 5
        self.linewidth_pt = 1.5

    def draw_plot(self, randomized_samples, mean, Cov):
        n_samples = len(randomized_samples)

        assert mean.ndim == 1
        dim = len(mean)
        assert Cov.ndim == 2
        assert Cov.shape == (dim, dim)
        assert np.array_equal(Cov, Cov.T)
        assert all(np.diag(Cov) > 0.0)

        steps = np.array([int(n) for n in np.linspace(0, n_samples, 101)])[1:]
        n_steps = len(steps)

        mean_hat_n = np.full([n_steps, dim], np.nan, float)
        Cov_hat_n = np.full([n_steps, dim, dim], np.nan, float)
        for i, n in enumerate(steps):
            mean_hat_n[i, :],  Cov_hat_n[i, :, :] = \
                estimate_sample_statistics(randomized_samples[:n])

        self.clear()

        self.suptitle("2D Monte Carlo Convergence Check",
                      fontsize=self.fontsize_pt)

        for i in [0, 1]:
            # Mean values
            subp = self.add_subplot(2, 5, i + 1)
            subp.set_title(rf"$X_{i+1}$", fontsize=self.fontsize_pt)
            subp.plot(steps, mean_hat_n[:, i],
                      'k.', markersize=self.markersize_pt)
            subp.axhline(mean[i], label=r"$\mathbb{E}$",
                         linestyle="--", linewidth=self.linewidth_pt,
                         color="red")
            subp.set_ylabel(r"$\hat{\mu}_n$", fontsize=self.fontsize_pt)
            subp.legend(loc="best", fontsize=self.fontsize_pt)
            subp.grid(True)

            subp = self.add_subplot(2, 5, i + 6)
            subp.semilogy(steps, np.fabs(1.0 - mean_hat_n[:, i] / mean[i]),
                          'k.', markersize=self.markersize_pt)
            subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
            subp.set_ylabel(r"$\hat{\mu}_n$ Relative Error",
                            fontsize=self.fontsize_pt)
            subp.grid(True)

            # Variances
            subp = self.add_subplot(2, 5, i + 3)
            subp.set_title(rf"$X_{i+1}$", fontsize=self.fontsize_pt)
            subp.plot(steps, Cov_hat_n[:, i, i],
                      'k.', markersize=self.markersize_pt)
            subp.axhline(Cov[i, i], label=r"$\mathbb{V}$",
                         linestyle="--", linewidth=self.linewidth_pt,
                         color="red")
            subp.set_ylabel(r"$\hat{\sigma^2}_n$", fontsize=self.fontsize_pt)
            subp.legend(loc="best", fontsize=self.fontsize_pt)
            subp.grid(True)

            subp = self.add_subplot(2, 5, i + 8)
            subp.semilogy(steps, np.fabs(1.0 - Cov_hat_n[:, i, i] / Cov[i, i]),
                          'k.', markersize=self.markersize_pt)
            subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
            subp.set_ylabel(r"$\hat{\sigma^2}_n$ Relative Error",
                            fontsize=self.fontsize_pt)
            subp.grid(True)

        # Covariance
        if Cov[0, 1] <= 1.0e-3:
            cov_err = np.fabs(Cov_hat_n[:, 0, 1] - Cov[0, 1])
            cov_err_label = "Absolute Error"
        else:
            cov_err = np.fabs(1.0 - Cov_hat_n[:, 0, 1] / Cov[0, 1])
            cov_err_label = "Relative Error"

        subp = self.add_subplot(2, 5, 5)
        subp.set_title(rf"$X_1 X_2$", fontsize=self.fontsize_pt)
        subp.plot(steps, Cov_hat_n[:, 0, 1],
                  'k.', markersize=self.markersize_pt)
        subp.axhline(Cov[0, 1], label=r"$Cov$",
                     linestyle="--", linewidth=self.linewidth_pt,
                     color="red")
        subp.set_ylabel(r"$\hat{Cov}_n$", fontsize=self.fontsize_pt)
        subp.legend(loc="best", fontsize=self.fontsize_pt)
        subp.grid(True)

        subp = self.add_subplot(2, 5, 10)
        subp.semilogy(steps, cov_err,
                      'k.', markersize=self.markersize_pt)
        subp.set_xlabel(r"N samples ($n$)", fontsize=self.fontsize_pt)
        subp.set_ylabel(rf"$\hat{{Cov}}_n$ {cov_err_label}",
                        fontsize=self.fontsize_pt)
        subp.grid(True)

        self.tight_layout()
