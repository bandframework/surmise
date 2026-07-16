import os
import sys
import json
import shutil
import unittest
import functools

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from create_scipy_stats_rng import create_scipy_stats_rng
from create_distribution import create_distribution
from create_sampler import create_sampler
from save_mcmc_results import save_mcmc_results
from load_mcmc_results import load_mcmc_results
from print_sample_statistics import print_sample_statistics
from MplMcmcApprox1D import MplMcmcApprox1D
from MplMcConvergence1D import MplMcConvergence1D
from MplMcConvergence2D import MplMcConvergence2D
from MplMcConvergenceMeansND import MplMcConvergenceMeansND
from MplMcConvergenceVarND import MplMcConvergenceVarND
from MplMcConvergenceCovND import MplMcConvergenceCovND
from MplCornerPlot import MplCornerPlot


class TestSampler(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        It is unusual for a derived TestCase to have its own constructor, let
        alone one that accepts an argument.  This is due to the desire to be
        able to run this test case within a command line script that accepts a
        test suite declaration file and creates a test case object for running
        on that suite.

        To manage this nonstandard design, the command line script assumes that
        it need only run the ``testAllSetups`` method.  Therefore, do *not* add
        any other ``test*`` methods to this test case.

        .. todo::
            * If test_spec keyword argument is not provided, then we should use
              a dedicated, default test suite declaration file collocated with
              this file.  This would be useful if we can get this integrated
              into the package's test infrastructure.
        """
        self.__fname_json = kwargs["test_spec"]
        del kwargs["test_spec"]

        super().__init__(*args, **kwargs)

        with open(self.__fname_json, "r") as fptr:
            self.__problems = json.load(fptr)

    def setUp(self):
        # We don't remove this folder in tearDown() since users might want to
        # inspect the results manually or use them as benchmarks for later
        # testing.
        self.__dir = Path().cwd().joinpath("SamplerTestResults")
        if self.__dir.is_file():
            os.remove(self.__dir)
        elif self.__dir.is_dir():
            shutil.rmtree(self.__dir)
        os.mkdir(self.__dir)

    def testAllSetups(self):
        for problem_name, problem in self.__problems.items():
            target_cfg = problem["TargetDistribution"]
            target_name = target_cfg["Name"]
            print()
            print(f"{target_name} Target Distribution Tests")
            print("=" * 80)
            target_distribution = create_distribution(target_cfg)

            for setup_name, test_setup in problem["TestSetups"].items():
                print()
                print(setup_name)
                print("-" * 45)

                sampler_name = test_setup["Sampler"]["Name"]
                name = f"{sampler_name}_{problem_name}_{setup_name}"
                self.__testSampler(name, target_distribution, test_setup)

    def __testSampler(self, name, target_distribution, test_setup):
        # ----- ESTHETICS
        FONTSIZE = 12
        MARKERSIZE = 2
        LINEWIDTH = 2.0

        plt.style.use("ggplot")

        # ----  "HARDCODED"
        FNAME_H5 = self.__dir.joinpath(f"{name}.h5")

        # For quantile-quantile tables
        QUANTILES_PROBS = np.array([0.01, 0.05, 0.1, 0.25,
                                    0.5, 0.75, 0.9, 0.95, 0.99])
        # For marginal histograms in corner plots
        PLOT_QUANTILES_PROB = np.array([0.1, 0.5, 0.9])
        # N points for evaluating target pdf in corner plots
        GRID_SIZE = 500

        # ----- TRUE MOMENTS
        dimension = target_distribution.dimension
        mu_true, var_true = target_distribution.moments

        # ----- MCMC CONFIGURATION
        rng_cfg = test_setup["rng"]

        # Starting distribution
        start_distribution = None
        if "StartDistribution" in test_setup:
            start_cfg = test_setup["StartDistribution"]
            start_name = start_cfg["Name"]
            print(f"Start distribution\t{start_name}")
            start_distribution = create_distribution(start_cfg)

        # -- Create sampler & load sampler-specific configuration
        sampler_name, run_MCMC, sampler_cfg = create_sampler(test_setup)
        scipy_stats_rng = create_scipy_stats_rng(rng_cfg)

        n_samples = sampler_cfg["numsamp"]

        # ------ RUN SAMPLER & CONFIRM REASONABLE RESULTS
        print()
        print("Sampling ...\t\t", end="")
        sys.stdout.flush()
        # TODO: Should we really pass an RNG as an argument?  We are creating a
        # function that is storing the RNG internally, which seems bad.  Also,
        # it seems like the sampler should be in charge of the RNG and pass it
        # to draw_func directly in the way that it deems best.  That way the RNG
        # use is happening all in one place.
        #
        # Seems like the interface of draw_func should be updated so that it
        # accepts an RNG on all calls.
        #
        # Since it's an argument passed to the sampler, should calling code be
        # allowed to set the RNG into it however they please?  Should surmise
        # define these so that they always get the surmise-wide RNG in use at
        # the moment if it is sampling?
        #
        # What if it's a sampler function that users pass in?  We can't force it
        # to use the surmise RNG?
        start_dist_sampler = None
        if start_distribution is not None:
            start_dist_sampler = functools.partial(start_distribution.sample,
                                                   rng=scipy_stats_rng)
        result_1 = run_MCMC(
            logpost_func=target_distribution.logpdf,
            draw_func=start_dist_sampler,
            scipy_stats_rng=scipy_stats_rng
        )
        self.assertFalse(FNAME_H5.exists())
        save_mcmc_results(FNAME_H5, sampler_name, result_1)
        self.assertTrue(FNAME_H5.is_file())
        print("done")
        sys.stdout.flush()

        samples = result_1["theta"]
        self.assertEqual(len(samples), n_samples)

        sample_skip = test_setup["SampleSkip"]
        samples = samples[::sample_skip]

        # -- Log sample statistics
        print()
        print_sample_statistics(target_distribution, samples)
        print()

        # -- Compute distribution approximation quality info
        quantiles_true = target_distribution.inv_cdf(QUANTILES_PROBS)
        if dimension == 1:
            quantiles_results = np.quantile(samples, QUANTILES_PROBS)
            quantiles_absdiff = np.abs(quantiles_true - quantiles_results)
            table_quantiles = np.column_stack(
                (QUANTILES_PROBS,
                 quantiles_true, quantiles_results,
                 quantiles_absdiff))
            print(['Prob.', 'True Quantiles', 'Sample Quantiles', 'Abs. Diff.'])
            print(table_quantiles)
        elif dimension == 2:
            quantiles_results = np.atleast_2d(
                np.quantile(samples, QUANTILES_PROBS, axis=0))
            quantiles_absdiff = np.abs(quantiles_true.T - quantiles_results)

            table_quantiles = np.column_stack(
                (np.atleast_2d(QUANTILES_PROBS).T,
                 quantiles_absdiff))
            print(['Prob.', 'Abs. Diffs. in Quantiles (each dim.)'])
            print(table_quantiles)

        # -- Visualize results
        if test_setup["Plot"]:
            # Randomly shuffle the original MCMC samples so that the integrated
            # quantity convergence plots mimic what we would see if the samples
            # used to approximate the integrals were drawn independently.
            resampling = scipy_stats_rng.choice(
                np.arange(len(samples)),
                size=len(samples),
                replace=False
            )

            if dimension == 1:
                from statsmodels.graphics.tsaplots import plot_acf
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
                plot_acf(result_1['theta'], ax=ax[0], lags=25,
                         title='Before subsampling')
                ax[0].set_xlabel('lags')

                # Subsampling
                plot_acf(samples, ax=ax[1], lags=25, title='Subsampled')
                ax[1].set_xlabel('lags')
                plt.tight_layout()

                fig = plt.figure(num=2, FigureClass=MplMcConvergence1D,
                                 figsize=(8, 5))
                fig.fontsize_pt = FONTSIZE
                fig.markersize_pt = MARKERSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(samples[resampling], mu_true, var_true)

                fig = plt.figure(num=3, FigureClass=MplMcmcApprox1D,
                                 figsize=(10, 4))
                fig.fontsize_pt = FONTSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(target_distribution, start_distribution,
                              samples, 0.05)
            elif dimension == 2:
                corner_bins = test_setup["CornerPlotBins"]

                fig = plt.figure(num=1, FigureClass=MplMcConvergence2D,
                                 figsize=(12, 5))
                fig.fontsize_pt = FONTSIZE
                fig.markersize_pt = MARKERSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(samples[resampling], mu_true, var_true)

                fig = plt.figure(num=2, FigureClass=MplCornerPlot,
                                 figsize=(8, 8))
                fig.alpha = 0.7
                fig.fontsize_pt = FONTSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(target_distribution, samples,
                              PLOT_QUANTILES_PROB, GRID_SIZE, corner_bins)
            elif dimension in [3, 4]:
                corner_bins = test_setup["CornerPlotBins"]

                fig = plt.figure(num=1, FigureClass=MplMcConvergenceMeansND,
                                 figsize=(12, 5))
                fig.fontsize_pt = FONTSIZE
                fig.markersize_pt = MARKERSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(samples[resampling], mu_true)

                fig = plt.figure(num=2, FigureClass=MplMcConvergenceVarND,
                                 figsize=(12, 5))
                fig.fontsize_pt = FONTSIZE
                fig.markersize_pt = MARKERSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(samples[resampling], var_true)

                fig = plt.figure(num=3, FigureClass=MplMcConvergenceCovND,
                                 figsize=(12, 5))
                fig.fontsize_pt = FONTSIZE
                fig.markersize_pt = MARKERSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(samples[resampling], var_true)

                fig = plt.figure(num=4, FigureClass=MplCornerPlot,
                                 figsize=(8, 8))
                fig.alpha = 0.7
                fig.fontsize_pt = FONTSIZE
                fig.linewidth_pt = LINEWIDTH
                fig.draw_plot(target_distribution, samples,
                              PLOT_QUANTILES_PROB, GRID_SIZE, corner_bins)
            else:
                # TODO: Too many integrated quantities to include a full set of
                # convergence plots.  Don't plot at all or plot a subset of the
                # quantities with the worst convergence?
                raise NotImplementedError("Only 1D to 4D visualizations for now")
            plt.show()

        # TODO: Compute effective N samples
        # TODO: Compute Rhat
        # TODO: Check against CDF?
        # TODO: Automatic check on statistical estimates?

        # ----- CHECK IDENTICAL TO BENCHMARK
        fname_benchmark = test_setup["Benchmark"]
        self.assertTrue(isinstance(fname_benchmark, str))
        if fname_benchmark != "":
            fname_benchmark = Path(fname_benchmark).resolve()
            self.assertTrue(fname_benchmark.is_file())
            self.__compare_results(sampler_name, fname_benchmark, FNAME_H5)

        # ----- CONFIRM DETERMINISTIC
        # Rerun with identical RNG setup & confirm bitwise exact samples
        scipy_stats_rng = create_scipy_stats_rng(rng_cfg)
        print()
        print("Sampling again ...\t", end="")
        sys.stdout.flush()
        start_dist_sampler = None
        if start_distribution is not None:
            start_dist_sampler = functools.partial(start_distribution.sample,
                                                   rng=scipy_stats_rng)
        result_2 = run_MCMC(
            logpost_func=target_distribution.logpdf,
            draw_func=start_dist_sampler,
            scipy_stats_rng=scipy_stats_rng
        )
        print("done")
        sys.stdout.flush()

        self.__compare_sampler_specific(sampler_name, result_1, result_2)
        theta_1 = result_1["theta"]
        theta_2 = result_2["theta"]
        self.assertTrue(
            np.array_equal(theta_1, theta_2, equal_nan=False)
        )

    def __compare_sampler_specific(self, sampler_name, benchmark, new):
        if sampler_name.lower() == "metropolis_hastings":
            # TODO: lpostlist should probably be saved in the files so that we
            # can check new results against benchmarks as well.
            # self.assertEqual(set(benchmark), {"theta", "acc_rate", "lpostlist"})
            self.assertEqual(new["acc_rate"], benchmark["acc_rate"])
            self.assertTrue(0.3 <= benchmark["acc_rate"] <= 0.4)
        elif sampler_name.upper() == "LMC":
            # Nothing extra to test
            pass
        elif sampler_name.upper() == "PTLMC":
            # Nothing extra to test
            pass
        else:
            raise ValueError("Not testing sampler-specific results")
        self.assertEqual(set(benchmark), set(new))

    def __compare_results(self, sampler_name, fname_benchmark, fname_new):
        print()
        print("Regression Check")
        print(f"New\t\t\t{fname_new}")
        print(f"Benchmark\t\t{fname_benchmark}")
        benchmark = load_mcmc_results(fname_benchmark)
        new = load_mcmc_results(fname_new)

        self.__compare_sampler_specific(sampler_name, benchmark, new)

        theta_new = new["theta"]
        theta_benchmark = benchmark["theta"]
        self.assertTrue(
            np.array_equal(theta_new, theta_benchmark, equal_nan=False)
        )
