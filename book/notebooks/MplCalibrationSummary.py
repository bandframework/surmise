import matplotlib.figure as mfig


class MplCalibrationSummary(mfig.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__fontsize = -1
        self.__tick_fontsize = -1
        self.__linewidth = -1

    @property
    def fontsize(self):
        return self.__fontsize

    @fontsize.setter
    def fontsize(self, size_points):
        assert size_points > 0
        self.__fontsize = size_points

    @property
    def tick_fontsize(self):
        return self.__tick_fontsize

    @tick_fontsize.setter
    def tick_fontsize(self, size_points):
        assert size_points > 0
        self.__tick_fontsize = size_points

    @property
    def linewidth(self):
        return self.__linewidth

    @linewidth.setter
    def linewidth(self, width_points):
        assert width_points > 0
        self.__linewidth = width_points

    def draw_figure(self, parameter, samples_MC, samples_randomized, bins):
        assert self.fontsize > 0
        assert self.tick_fontsize > 0
        assert self.linewidth > 0

        assert len(samples_MC) == len(samples_randomized)

        self.clear()

        subp_1 = self.add_subplot(221)
        subp_1.set_title("MC Order", fontsize=self.fontsize)
        subp_1.plot(samples_MC, 'k-', linewidth=self.linewidth)
        subp_1.set_xlabel("Calibration Sample", fontsize=self.fontsize)
        subp_1.set_ylabel(f"{parameter} Sample", fontsize=self.fontsize)
        subp_1.tick_params(axis="both", labelsize=self.tick_fontsize)
        subp_1.grid(True)

        subp = self.add_subplot(222, sharex=subp_1, sharey=subp_1)
        subp.set_title("Randomized Order", fontsize=self.fontsize)
        subp.plot(samples_randomized, 'k-', linewidth=self.linewidth)
        subp.set_xlabel("Calibration Sample", fontsize=self.fontsize)
        subp.set_ylabel(f"{parameter} Sample", fontsize=self.fontsize)
        subp.tick_params(axis="both", labelsize=self.tick_fontsize)
        subp.grid(True)

        subp = self.add_subplot(223, sharey=subp_1)
        subp.boxplot(samples_MC)
        subp.set_ylabel(f"{parameter} Sample", fontsize=self.fontsize)
        subp.tick_params(axis="x", label1On=False)
        subp.tick_params(axis="y", labelsize=self.tick_fontsize)
        subp.grid(False)

        subp = self.add_subplot(224)
        subp.hist(samples_MC, bins=bins, density=True)
        subp.set_xlabel(f"{parameter} Sample", fontsize=self.fontsize)
        subp.set_ylabel("Probability Density", fontsize=self.fontsize)
        subp.tick_params(axis="both", labelsize=self.tick_fontsize)
        subp.grid(True)
