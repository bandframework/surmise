import corner

import numpy as np
import matplotlib.figure as mfig

from matplotlib.lines import Line2D


class MplCornerPlot(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = 0.7
        self.fontsize_pt = 18
        self.linewidth_pt = 1.5

    def draw_plot(self, target_distribution, samples, quantiles,
                  grid_size, bins):
        dimension = target_distribution.dimension
        plot_quantiles_true = target_distribution.inv_cdf(quantiles)
        assert plot_quantiles_true.ndim == 2
        assert plot_quantiles_true.shape == (dimension, len(quantiles))

        # Defining ranges
        lo = samples.min(axis=0)
        hi = samples.max(axis=0)
        pad = 0.10 * (hi - lo)
        ranges = np.array((lo - pad, hi + pad)).T

        self.clear()

        corner.corner(
            samples,
            bins=bins,
            range=ranges,
            quantiles=quantiles,
            show_titles=True,
            title_fmt=".3g",
            hist_kwargs={'linewidth': self.linewidth_pt,
                         'density': True},
            fig=self,
            # TODO: Moses to decide on setting over normal test case
            # plot_contour=True,
        )

        # Add quantiles and target distribution pdf on diagonals
        axes = np.array(self.axes).reshape((dimension, dimension))
        for d in range(dimension):
            # We are assuming that this is ordered according to the column
            # ordering of samples and that the column ordering of samples
            # correctly matches the ordering used to construct the target
            # distribution.
            ax = axes[d, d]

            # Plot marginal pdf
            x = np.linspace(*ranges[d], grid_size)
            try:
                marginal_pdf = target_distribution.marginal_pdf(x, d)
            except Exception:
                marginal_pdf = None
            if marginal_pdf is not None:
                ax.plot(x, marginal_pdf, color='C0', ls=":",
                        lw=self.linewidth_pt, alpha=self.alpha)

            added_truth = False
            for q in plot_quantiles_true[d, :]:
                ax.axvline(q, color="C0", ls=":",
                           lw=self.linewidth_pt, alpha=self.alpha,
                           label='Truth' if not added_truth else None)
                added_truth = True

        # Add target distribution on off-upper diagonal
        sample_median = np.median(samples, axis=0)
        for i in range(dimension):
            for j in range(i + 1, dimension):
                # TODO: Moses to figure out plot grid aesthetics
                ax = axes[i, j]

                # TODO: If we want to use this for 3D or higher, does this code
                # need updating?
                X, Y = np.meshgrid(
                    np.linspace(*ranges[j], grid_size),
                    np.linspace(*ranges[i], grid_size)
                )
                points = np.tile(sample_median, (X.size, 1))
                points[:, j] = X.ravel()
                points[:, i] = Y.ravel()
                target_pdf = target_distribution.pdf(points).reshape(X.shape)

                pdf_extent = [np.min(points[:, j]), np.max(points[:, j]),
                              np.min(points[:, i]), np.max(points[:, i])]

                ax.imshow(target_pdf, interpolation="none", aspect="auto",
                          origin="lower", extent=pdf_extent)
                ax.set_xlim(axes[j, i].get_ylim())
                ax.set_ylim(axes[j, i].get_xlim())

                # TODO: Is there a routine for computing the contours
                # corresponding to the different quantiles?
                # ax.contour(
                #     X, Y, target_pdf,
                #     levels=5,
                #     colors="C0",
                #     linestyles=":",
                #     linewidths=self.linewidth_pt
                # )

        # Producing legend
        handles = [
            Line2D([0], [0],
                   color="C0", ls=":", lw=self.linewidth_pt, alpha=self.alpha,
                   label="Truth"),
            Line2D([0], [0],
                   color="k", ls="--", lw=self.linewidth_pt, alpha=self.alpha,
                   label="Empirical"),
        ]

        self.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )

        self.tight_layout()
