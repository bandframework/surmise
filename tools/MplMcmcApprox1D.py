import numpy as np
import matplotlib.figure as mfig


class MplMcmcApprox1D(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fontsize_pt = 16
        self.linewidth_pt = 1.5

    def draw_plot(self, target_distribution, start_distribution,
                  theta_samples, alpha):
        min_theta = np.min(theta_samples)
        max_theta = np.max(theta_samples)
        theta_data = np.linspace(
            (1.0 + alpha) * min_theta - alpha * max_theta,
            (1.0 + alpha) * max_theta - alpha * min_theta,
            1000
        )
        target_pdf = target_distribution.pdf(theta_data)

        self.clear()

        self.suptitle("MCMC 1D Target Distribution Approximation",
                      fontsize=self.fontsize_pt)

        subp = self.add_subplot(121)
        if start_distribution is not None:
            start_pdf = start_distribution.pdf(theta_data)
            subp.plot(theta_data, start_pdf, "k-",
                      linewidth=self.linewidth_pt, label="Start PDF")
        subp.hist(theta_samples, density=True, bins="auto")
        subp.set_xlabel(r"$\theta$", fontsize=self.fontsize_pt)
        subp.set_ylabel("PDF", fontsize=self.fontsize_pt)
        subp.legend(loc="best", fontsize=self.fontsize_pt)
        subp.grid(True)

        subp = self.add_subplot(122, sharex=subp)
        subp.plot(theta_data, target_pdf, "k-",
                  linewidth=self.linewidth_pt, label="Target PDF")
        subp.hist(theta_samples, density=True, bins="auto")
        subp.set_xlabel(r"$\theta$", fontsize=self.fontsize_pt)
        subp.set_ylabel("PDF", fontsize=self.fontsize_pt)
        subp.legend(loc="best", fontsize=self.fontsize_pt)
        subp.grid(True)

        self.tight_layout()
