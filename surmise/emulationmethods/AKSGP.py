#!/usr/bin/env python3
'''
Created Aug 2024

@author: Sunil Jaiswal (jaiswal.61@osu.edu)

Reference:
    This implementation is based on the methodology proposed in the following paper:
    Sunil Jaiswal et al., "Title of the Paper," arXiv:xxxx.xxxxx, Year.
    Available at: https://arxiv.org/abs/xxxx.xxxxx

Note:
    This file can be used as a standalone module for GPR without additional dependencies from the SURMISE package.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, DotProduct, RationalQuadratic, ExpSineSquared, Kernel, Product
from sklearn.gaussian_process import GaussianProcessRegressor
from joblib import Parallel, delayed
import logging
import time
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Function to define the kernels as a dictionary
def get_kernels(input_dim):
    """
    Returns a dictionary of Gaussian Process kernels (instances of `sklearn.gaussian_process.kernels.Kernel`) 
    with names as keys and corresponding kernel objects as values. 
    These kernels are availaible for training and can be extended by adding more combinations.

    The kernels are designed to accommodate different modeling needs by combining anisotropic kernels with 
    isotropic (stationary and non-stationary) kernels in various ways.

    Considered Kernels:
    - **Anisotropic and Stationary Base Kernels:**
        - `Matern12`: Matern kernel with nu=0.5. Equivalent to the exponential kernel.
        - `Matern32`: Matern kernel with nu=1.5.
        - `Matern52`: Matern kernel with nu=2.5.
        - `RBF`: Radial Basis Function (Gaussian) kernel.
    
    - **Isotropic Kernels (Used in Combination Only):**
        - `DotProduct`: Non-stationary kernel, useful for modeling linear trends.
        - `ExpSineSquared`: Stationary kernel, ideal for periodic data.
        - `RationalQuadratic`: Stationary kernel, a scale mixture of RBF kernels, useful for varying smoothness.

    Included Kernels:
    - **Base Kernels (Anisotropic and Stationary):**
        - `Matern12`, `Matern32`, `Matern52`, `RBF`.
    
    - **Kernel Combinations:**
        - Each isotropic kernel is combined with the base kernels through addition (`+`) and multiplication (`*`) to 
          capture different interactions between the features:
            - `DotProduct + Matern12`, `DotProduct * Matern12`, etc.
            - `ExpSineSquared + Matern32`, `ExpSineSquared * Matern32`, etc.
            - `RationalQuadratic + RBF`, `RationalQuadratic * RBF`, etc.

    Parameters:
        input_dim (int): The dimensionality of the input space used to define the kernel length scales.

    Returns:
        kernel_dict (dict): A dictionary where keys are kernel names (str) and values are kernel objects.
    """

    lb, ub = 1e-3, 1e3  # lower bound, upper bound of input space. Note that the input space is standardized.

    kernel_dict = {
        'Matern12': 1.0 * Matern(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(lb, ub), nu=0.5),
        'Matern32': 1.0 * Matern(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(lb, ub), nu=1.5),
        'Matern52': 1.0 * Matern(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(lb, ub), nu=2.5),
        'RBF': 1.0 * RBF(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(lb, ub)),
        # 
        # 'DotProduct+Matern12': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) + 
        #                         1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=0.5),
        # 'DotProduct+Matern32': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) + 
        #                         1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=1.5),
        # 'DotProduct+Matern52': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) + 
        #                         1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=2.5),
        # 'DotProduct+RBF': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) + 
        #                     1.0 * RBF(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub)),
        # 'DotProduct*Matern12': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) * 
        #                         Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=0.5),
        # 'DotProduct*Matern32': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) * 
        #                         Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=1.5),
        # 'DotProduct*Matern52': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) * 
        #                          Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=2.5),
        # 'DotProduct*RBF': 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(lb, ub)) * 
        #                     RBF(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub)),
        # # 
        # 'ExpSineSquared+Matern12': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) + 
        #                             1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=0.5),
        # 'ExpSineSquared+Matern32': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) + 
        #                             1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=1.5),
        # 'ExpSineSquared+Matern52': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) + 
        #                             1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=2.5),
        # 'ExpSineSquared+RBF': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) + 
        #                         1.0 * RBF(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub)),
        # 'ExpSineSquared*Matern12': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) * 
        #                              Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=0.5),
        # 'ExpSineSquared*Matern32': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) * 
        #                              Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=1.5),
        # 'ExpSineSquared*Matern52': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) * 
        #                              Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=2.5),
        # 'ExpSineSquared*RBF': 1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0) * 
        #                          RBF(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub)),
        # # 
        # 'RationalQuadratic+Matern12': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) + 
        #                                 1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=0.5),
        # 'RationalQuadratic+Matern32': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) + 
        #                                 1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=1.5),
        # 'RationalQuadratic+Matern52': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) + 
        #                                 1.0 * Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=2.5),
        # 'RationalQuadratic+RBF': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) + 
        #                             1.0 * RBF(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub)),
        # 'RationalQuadratic*Matern12': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) * 
        #                                 Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=0.5),
        # 'RationalQuadratic*Matern32': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) * 
        #                                 Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=1.5),
        # 'RationalQuadratic*Matern52': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) * 
        #                                 Matern(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub), nu=2.5),
        # 'RationalQuadratic*RBF': 1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0) * 
        #                             RBF(length_scale=np.ones(input_dim) / 2.0, length_scale_bounds=(lb, ub)),
        }
    
    return kernel_dict

# Function to define the distance metrics as a dictionary
def get_metrics():
    """
    Returns a dictionary of metric functions used to select the best performimg kernel for Gaussian Process regression.

    Returns:
        metric_list (dict): A dictionary where keys are metric names (str) and values are functions that compute the 
                            metric for Gaussian distributions given the means and standard deviations.
    """
    
    def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
        return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2.0 * sigma2**2) - 0.5

    def hellinger_distance_gaussian(mu1, sigma1, mu2, sigma2):
        term1 = np.sqrt(2.0 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
        term2 = np.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
        return np.sqrt(1.0 - term1 * term2)

    def wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2):
        return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)

    metric_list = {
        'KL Divergence': kl_divergence_gaussian,
        'Hellinger Distance': hellinger_distance_gaussian,
        'Wasserstein Distance': wasserstein_distance_gaussian
        }

    return metric_list


########################################################################################################
# Emulator class to fit a Gaussian Process and predict from it
class Emulator:
    def __init__(self, X, Y_mean, Y_std):
        """
        A class for performing Gaussian Process regression.

        The Emulator class is designed to handle multi-dimensional input and output data by training individual Gaussian 
        Process models for each output dimension. It supports automatic kernel selection (AKS) based on performance metrics.
    
        What it needs:
            - Valid numeric input features (`X`), target mean values (`Y_mean`), and target standard deviations (`Y_std`).
            - GP Kernels and metrics defined through the `get_kernels` and `get_metrics` functions.

        Key Methods:
            - `fit(kernel, nrestarts, n_jobs, seed)`: Trains GP models for each output dimension, 
                                                      with optional (default) automatic kernel selection method.
            - `predict(X_new, return_full_covmat)`: Predicts mean and uncertainty for new input data using the fitted GPs.

        Example usage: 
            >>> from sklearn.datasets import make_friedman2
            >>> X, y = make_friedman2(n_samples=100, noise=0.5, random_state=0)
            >>> emu = Emulator(X=X, Y_mean=y, Y_std=None)
            >>> emu.fit(kernel='AKS', nrestarts=10, n_jobs=-1, seed=42)
            >>> # Predict and compare with test data --->
            >>> Xtest, ytest = make_friedman2(n_samples=10, noise=0.5, random_state=0)
            >>> GP_means, GP_std = emu.predict(Xtest, return_full_covmat=False)
            >>> print("% error in mean:\n", (1.0 - GP_means/ytest.reshape(-1, 1))*100)
            
        Parameters:
            X (array-like): Input features of shape (n_samples, n_features).
            Y_mean (array-like): Mean values of the training data of shape (n_samples, n_outputs).
            Y_std (array-like): Standard deviation of the training data of shape (n_samples, n_outputs).
                                If 'Y_std = None' the method will treat Y_std as nugget during training.
            
        Raises:
            ValueError: 
                - If the arrays `X`, `Y_mean`, and `Y_std` contains NaN, inf, or any non-numeric values.
                - If the number of training points in `X` and `Y_mean` do not match.
                - If the shapes of `Y_mean` and `Y_std` do not match.
        """

        # Validate the arrays
        self._validate_array(X, 'X')
        self._validate_array(Y_mean, 'Y_mean')
        
        if Y_std is None:
            # Adding nugget as Y_std:
            column_wise_mean = np.mean(Y_mean, axis=0)
            Y_std = np.random.uniform(1e-4, 1e-8, Y_mean.shape) * column_wise_mean # Adding nugget
        else:
            self._validate_array(Y_std, 'Y_std')

        # If X, Y_mean or Y_std are 1D, reshape them to be 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y_mean.ndim == 1:
            Y_mean = Y_mean.reshape(-1, 1)
        if Y_std.ndim == 1:
            Y_std = Y_std.reshape(-1, 1)
        
        # Ensure that the number of training points in X and Y_mean match
        if X.shape[0] != Y_mean.shape[0]:
            raise ValueError("Number of training points in 'X' and 'Y_mean' must match.")
            
        # Ensure that the shapes of Y_mean and Y_std match
        if Y_mean.shape != Y_std.shape:
            raise ValueError("Shapes of 'Y_mean' and 'Y_std' must match.")
        
        self.X = X
        self.Y_mean = Y_mean
        self.Y_std = Y_std

        self._scaler_X = StandardScaler()
        self._scaler_Y = StandardScaler()

        self.gps = []

        self.trainwallclocktime = []
        self.traintotalcputime = []
        self.predictwallclocktime = []
        self.predicttotalcputime = []

        self.selected_kernels = []
        # Initialize the dictionaries of kernels and metrics for best kernel selction 
        input_dim = self.X.shape[1]  # dimensionality of input space
        
        # Initialize the kernel and metric dictionaries
        self.kernels_list = get_kernels(input_dim)
        self.metrics = get_metrics()

        # Check if the kernel and metric dictionaries exist
        if not self.kernels_list or not self.metrics:
            raise ValueError("Kernel dict or metric dict is empty or not defined properly.")
    
    # ---------------------------------------------------------------------------------------------
    
    def fit(self, kernel: str = 'AKS', nrestarts: int = 10, n_jobs: int = -1, seed: int = None) -> None:

        """
        Train individual Gaussian Processes (GPs) for each output dimension.
        
        Parameters:
            kernel (str): The type of kernel to use for the Gaussian Process. Default is 'AKS' -- Automatic kernel selection. 
            List of kernels should be defined in "get_kernels()". Options include:
                - 'AKS' : The function will train GPs with all kernels defined in 'get_kernels()' and automatically select the best one.
                          The best kernel is chosen by fitting GPs (with 90% of training data) using each kernel and evaluating their 
                          performance using different distance metrics defined in 'get_metrics()' (KL divergence, Hellinger distance,
                          Wasserstein distance) on the rest 10% of the training data. 
                          See the "_select_best_kernels()" function for more details.
                          The GPs are then retrained with the selected kernels for each output dimension with all training data.
                - 'Matern12': Matern kernel with nu=0.5.
                - 'Matern32': Matern kernel with nu=1.5.
                - 'Matern52': Matern kernel with nu=2.5.
                - 'RBF': Radial Basis Function kernel.
                - ...
                
            nrestarts (int, optional): Number of restarts for the optimizer to improve convergence. Default is 10.
            
            n_jobs (int, optional): Number of CPU cores to use for parallel processing. 
                                    If set to -1, all available cores will be used. Default is -1.

            seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 42.
                          
        Raises:
            Exception: If an error occurs during the GP fitting process.
        """
        # Start wall-clock time
        start_wall_time = time.time()
        # Record the start CPU times
        start_cpu_times = psutil.cpu_times()

        if kernel=='AKS':
            #   - Split training data in 90% - 10% batch. 
            #   - Use 90% (training) batch to fit GPs for all output dimensions with all kernels in "kernel_list".
            #   - Use 10% (validation) batch to select best kernels for each output dimension.
            #   - Retrain the GPs with the selected kernels for each output dimension with all training data (before split).
            
            logger.info(
                f"Automatic kernel selection opted. Best kernel for each output dimension will be selected from the list of kernels:\n"
                f"   {list(self.kernels_list.keys())}\n"
            )
            
            try:
                # Split training data in: 90% (training) - 10% (validation) batch
                X_train, Ymean_train, Ystd_train, validation_X, validation_Ymean, validation_Ystd = self.split_train_validation(
                                                                                                            X=self.X, 
                                                                                                            Ymean=self.Y_mean, 
                                                                                                            Ystd=self.Y_std, 
                                                                                                            train_validation_ratio=0.9, 
                                                                                                            seed=seed)
            
                logger.info(f"Shape of training arrays: {X_train.shape}, {Ymean_train.shape}, {Ystd_train.shape}")
                logger.info(f"Shape of validation arrays: {validation_X.shape}, {validation_Ymean.shape}, {validation_Ystd.shape}")

                # Initializing GP dictionary to store the fitted GPs for all available kernels
                gplist = {}

                # Training GPs with different kernels for all output dimensions in parallel
                logger.info("Training GPs with all available kernels...")

                # Standardize data before GP training
                X_stnd, Ymean_stnd, Ystd_stnd = self._preprocess_data(X=X_train, Y_mean=Ymean_train, Y_std=Ystd_train)

                for kernel_name, ker in self.kernels_list.items():
                    gplist[kernel_name] = Parallel(n_jobs=n_jobs)(
                        delayed(self.fit_singleGP)(Xfit = X_stnd, 
                                                   Yfit_mean = sample_column, 
                                                   Yfit_std = Ystd_stnd[:, i], 
                                                   kernel = ker,
                                                   nrestarts = nrestarts
                                                  )
                        for i, sample_column in enumerate(Ymean_stnd.T)
                    )
                    logger.info(f"  Trained GPs with {kernel_name} kernels.")

                
                logger.info("Finding best kernels for each output dimension...")

                # First compute the metrics by comparing the fitted GPs with validation batch for all kernels
                metrics_result = self._compute_metrics(GP_dict = gplist,
                                                       X_val = validation_X, 
                                                       Ymean_val = validation_Ymean, 
                                                       Ystd_val = validation_Ystd
                                                      )

                # Now select the best kernel 
                best_kernels = self._select_best_kernels(metrics_result)
                self.selected_kernels = best_kernels # append to selected_kernels

                assert len(Ymean_stnd[1]) == len(best_kernels), (
                    "Error during GP fit: Number of best kernels not equal to number of output dimension."
                    "Check the _select_best_kernels function."
                )
                
                logger.info(f"  Selected best kernels for each output dimension:\n   {best_kernels}\n")
                
                del gplist  # Free memory of the gplist objects

                #  - Retrain the GPs with the selected kernels for each output dimension with all training data
                #  - Append the trained GPs to self.gps
                logger.info("Retraining the GPs with selected best kernels using all training data...")

                # Standardize data before GP retraining
                X_stnd, Ymean_stnd, Ystd_stnd = self._preprocess_data(X=self.X, Y_mean=self.Y_mean, Y_std=self.Y_std)

                self.gps = Parallel(n_jobs=n_jobs)(
                    delayed(self.fit_singleGP)(Xfit=X_stnd, 
                                               Yfit_mean=sample_column, 
                                               Yfit_std=Ystd_stnd[:, i], 
                                               kernel=self.kernels_list[best_kernels[i]],
                                               nrestarts = nrestarts
                                              )
                    for i, sample_column in enumerate(Ymean_stnd.T)
                )
                del best_kernels, metrics_result # Free memory
                
                logger.info("Retraining GPs complete.\n")
            
            except Exception as e:
                logger.error(f"Error during GP fit with automatic kernel selection: {e}")
                raise


        else:
            # Check if the specified kernel is valid and select the specified kernel
            if kernel not in self.kernels_list:
                raise ValueError(
                    f"Unsupported kernel type: {kernel}. Available kernels: {list(self.kernels_list.keys())}.\n" 
                    f"Or choose 'AKS' for Automatic Kernel Selection."
                )
                
            ker = self.kernels_list[kernel]
            self.selected_kernels = ker  # append to selected_kernels
    
            logger.info(f"Training GPs with {kernel} kernel...\n")
            try:
                # Standardize data before GP retraining
                X_stnd, Ymean_stnd, Ystd_stnd = self._preprocess_data(X=self.X, Y_mean=self.Y_mean, Y_std=self.Y_std)
                
                # Training GPs for different output dimension in parallel
                self.gps = Parallel(n_jobs=n_jobs)(
                    delayed(self.fit_singleGP)(Xfit=X_stnd, 
                                                 Yfit_mean=sample_column, 
                                                 Yfit_std=Ystd_stnd[:, i], 
                                                 kernel=ker, 
                                                 nrestarts=nrestarts
                                                )
                    for i, sample_column in enumerate(Ymean_stnd.T)
                )
            except Exception as e:
                logger.error(f"Error during GP fit: {e}")
                raise

        # Log the kernel information of the fitted GPs
        for i, sample_column in enumerate(Ymean_stnd.T):
            logger.info(
                    f"Kernel after GP training for output dimension {i}:\n{self.gps[i].kernel_}\n"
                    f"  Log-marginal-likelihood: {self.gps[i].log_marginal_likelihood_value_}\n"
                )
            
        # Record the end CPU times
        end_cpu_times = psutil.cpu_times()
        # End wall-clock time
        end_wall_time = time.time()
        
        # Calculate the total CPU time
        user_time = end_cpu_times.user - start_cpu_times.user
        system_time = end_cpu_times.system - start_cpu_times.system
        self.traintotalcputime = user_time + system_time
        
        # Calculate the wall-clock time
        self.trainwallclocktime = end_wall_time - start_wall_time

        del start_wall_time, start_cpu_times, end_cpu_times, user_time, system_time  # Free memory

    
    def predict(self, X_new, return_full_covmat=False):
        """
        Predicts from fitted GP at new input points.

        Parameters:
            X_new (array-like): predict input points.
            return_full_covmat (bool): Whether to return the full covariance matrix. Defaults to False.

        Returns:
            means (array-like): Predicted means in the original scale.
                - means[:, i] access means at the input points X_new for the i-th GP.
            std_devs (array-like) or covariances (array-like): Predicted standard deviations (default) 
                        or full covariance matrices in the original scale.
                - std_devs[:, i] or covariances[i] access for the i-th GP.
                
        Raises:
            ValueError: If dimensions of training input space does not match the dimension of new input points.
            Exception: If an error occurs during the GP fitting process.
        """

        # Start wall-clock time
        start_wall_time = time.time()
        # Record the start CPU times
        start_cpu_times = psutil.cpu_times()
        
        if  X_new.shape[1] != self.X.shape[1]:
            raise ValueError(
                    f"Error during GP predict: "
                    f"Dimensions of predict input space: {X_new.shape[1]} must match dimensions of training input space: {self.X.shape[1]}."
                )
        try:
            X_new_scaled = self._scaler_X.transform(X_new)
            predictions = [gp.predict(X_new_scaled, return_cov=True) for gp in self.gps]
            means_scaled, covariances_scaled = zip(*predictions) 
            
            # Transform the means back to the original scale
            means_scaled = np.column_stack(means_scaled)
            means = self._scaler_Y.inverse_transform(means_scaled)

            # Apply the scaling to all covariance matrices at once
            scale = self._scaler_Y.scale_
            covariances = [cov * scale[i] ** 2 for i, cov in enumerate(covariances_scaled)]

            if return_full_covmat:
                # Record the end CPU times
                end_cpu_times = psutil.cpu_times()
                # End wall-clock time
                end_wall_time = time.time()
                # Calculate the total CPU time
                user_time = end_cpu_times.user - start_cpu_times.user
                system_time = end_cpu_times.system - start_cpu_times.system
                self.predicttotalcputime = user_time + system_time
                # Calculate the wall-clock time
                self.predictwallclocktime = end_wall_time - start_wall_time

                del start_wall_time, start_cpu_times, end_cpu_times, user_time, system_time  # Free memory
        
                return means, covariances
                
            else:
                # Extract the diagonal (variance) from all covariance matrices at once
                variances = np.array([np.diag(cov) for cov in covariances])

                # Check if any variances are negative and set them to 0
                variances_negative = variances < 0
                if np.any(variances_negative):
                    logger.warning(
                        "Predicted variances contain negative values. Setting those variances to 0."
                    )
                    variances[variances_negative] = 0.0

                std_devs = np.sqrt(variances).T  # Transpose to match the expected output format

                # Record the end CPU times
                end_cpu_times = psutil.cpu_times()
                # End wall-clock time
                end_wall_time = time.time()
                # Calculate the total CPU time
                user_time = end_cpu_times.user - start_cpu_times.user
                system_time = end_cpu_times.system - start_cpu_times.system
                self.predicttotalcputime = user_time + system_time
                # Calculate the wall-clock time
                self.predictwallclocktime = end_wall_time - start_wall_time

                del start_wall_time, start_cpu_times, end_cpu_times, user_time, system_time  # Free memory

                return means, std_devs
        
        except Exception as e:
            logger.error(f"Error during GP predict: {e}")
            raise


# ====================================================================================================

    @staticmethod
    def _validate_array(arr, name):
        """
        Validates that the array contains only numeric values and has no NaN or inf values.

        Parameters:
            arr (array-like): The array to validate.
            name (str): The name of the array (used for error messages).
            
        Raises:
            ValueError: If the array contains non-numeric values, NaN, or inf.
        """
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"'{name}' must contain only numeric values.")
        if not np.isfinite(arr).all():
            raise ValueError(f"'{name}' contains NaN or inf values.")


    
    def _preprocess_data(self, X, Y_mean, Y_std):
        """
        Standardizes the training data by removing the mean and scaling to unit variance 
        across each variable in the input space and across each output dimension.

        Parameters:
            X (array-like): Input features.
            Y_mean (array-like): Mean values of the training data.
            Y_std (array-like): Standard deviation of the training data.
            
        Returns:
            scaled_X (array-like): Standardized input features.
            Z_mean (array-like): Standardized mean values of the target variable.
            Z_std (array-like): Standardized standard deviation of the target variable.
        
        Raises:
            Exception: If an error occurs during data preprocessing.
        """
        try:
            logger.info("  Standardizing input space...")
            scaled_X = self._scaler_X.fit_transform(X)
            
            logger.info("  Standardizing data...")
            Z_mean = self._scaler_Y.fit_transform(Y_mean)
            Z_std = Y_std / self._scaler_Y.scale_
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise

        #>>>>>>>>>> Check standardization :: start >>>>>>>>>>
        # Transform the means back to the original scale
        means_transformed_back = self._scaler_Y.inverse_transform(Z_mean)
        assert np.allclose(Y_mean, means_transformed_back, atol=1e-7), (
            "Error during data preprocessing: Original means and transformed back means do not match within tolerance:\n"
        )

        # Transform the standard deviations back to the original scale
        std_devs_transformed_back = Z_std * self._scaler_Y.scale_
        assert np.allclose(Y_std, std_devs_transformed_back, atol=1e-7), (
            "Error during data preprocessing: Original std deviations and transformed back std deviations do not match within tolerance:\n"
        )
        #<<<<<<<<<<<< Check standardization :: end <<<<<<<<<<<<

        return scaled_X, Z_mean, Z_std


    def fit_singleGP(self, Xfit, Yfit_mean, Yfit_std, kernel, nrestarts):
        """
        Fits a single Gaussian Process (GP) model using scikit-learn.
        
        Parameters:
            Xfit (array-like): Input features of shape (n_samples, n_features).
            Yfit_mean (array-like): Mean values of the training data of shape (n_samples, n_outputs).
            Yfit_std (array-like): Standard deviation of the training data of shape (n_samples, n_outputs).
            kernel (sklearn.gaussian_process.kernels.Kernel): Kernel object to be used by the GP.
            nrestarts (int): Number of restarts for the optimizer to enhance convergence.
            
        Returns:
            The fitted GP model.
        """
        gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=Yfit_std ** 2, 
            optimizer='fmin_l_bfgs_b', 
            n_restarts_optimizer=nrestarts
        )
        gp.fit(Xfit, Yfit_mean)
        
        return gp
        

    def split_train_validation(self, X, Ymean, Ystd, train_validation_ratio=0.9, seed=None):
        """
        Splits the dataset into training and validation sets based on the specified train_validation_ratio.
    
        Parameters:
            X (array-like): Input features of shape (n_samples, n_features).
            Ymean (array-like): Mean values of the training data of shape (n_samples, n_outputs).
            Ystd (array-like): Standard deviation of the training data of shape (n_samples, n_outputs).
            train_validation_ratio (float, optional): The ratio of the dataset to be used for training. Default is 0.9 (90%).
            seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 42.
    
        Returns:
            X_train (array-like): Training set of input features.
            Ymean_train (array-like): Training set of mean values.
            Ystd_train (array-like): Training set of standard deviations.
            X_val (array-like): Validation set of input features.
            Ymean_val (array-like): Validation set of mean values.
            Ystd_val (array-like): Validation set of standard deviations.
        """
        # Set the seed for reproducibility
        np.random.seed(seed)
    
        # Calculate the size of the training set as a percentage of the total data
        train_size = int(train_validation_ratio * X.shape[0])
        
        # Generate 'train_size' random unique indices
        selected_indices = np.random.choice(X.shape[0], size=train_size, replace=False)
        
        # Select the rows for training
        X_train = X[selected_indices]
        Ymean_train = Ymean[selected_indices]
        Ystd_train = Ystd[selected_indices]
        
        # Select the remaining rows for the validation set
        X_val = np.delete(X, selected_indices, axis=0)
        Ymean_val = np.delete(Ymean, selected_indices, axis=0)
        Ystd_val = np.delete(Ystd, selected_indices, axis=0)
    
        return X_train, Ymean_train, Ystd_train, X_val, Ymean_val, Ystd_val

    
    def _compute_metrics(self, GP_dict, X_val, Ymean_val, Ystd_val):
        """
        Computes and aggregates metric results for each kernel by comparing predicted GP means and stds with the validation data.
    
        Parameters:
            GP_dict (dict): Dictionary of fitted GPs for each kernel.
                - Key (str): The name of the kernel.
                - Value (list): List of GaussianProcessRegressor models fitted with the corresponding kernel.
                
            X_val (array-like): Input features of the validation data.
            Ymean_val (array-like): Mean values of the validation data.
            Ystd_val (array-like): Standard deviations of the validation data.

        
        Returns:
            dict: A nested dictionary containing the computed metrics for each kernel.
                - Outer Key (str): The name of the metric (e.g., 'KL Divergence', 'Hellinger Distance', 'Wasserstein Distance').
                - Outer Value (dict): 
                    - Inner Key (str): The name of the kernel (e.g., 'RBF', 'Matern52').
                    - Inner Value (array-like): The computed metric values for the validation data, with same shape as Ymean_val.
        """

        # Initialize the dictionary to store the results
        metric_results = {metric: {} for metric in self.metrics.keys()}
    
        # Iterate over each kernel and compute predictions and metrics
        for kernel_name in self.kernels_list.keys():
            self.gps = GP_dict[kernel_name]
            GP_means, GP_stds = self.predict(X_val, return_full_covmat=False)
    
            # Ensure that shapes of the arrays match
            assert GP_means.shape == GP_stds.shape == Ymean_val.shape == Ystd_val.shape, (
                "Error in _compute_metrics: Arrays of mean and std for metric computation must have the same shape."
            )
    
            # Compute the metrics for this kernel
            results = {}
            for metric_name, metric_func in self.metrics.items():
                results[metric_name] = metric_func(GP_means, GP_stds, Ymean_val, Ystd_val)
    
            # Store the results for the current kernel
            for metric_name, value in results.items():
                metric_results[metric_name][kernel_name] = value

        metric_results[metric_name][kernel_name]
        assert  metric_results[metric_name][kernel_name].shape == Ymean_val.shape, (
            "Error in _compute_metrics calculation: Shape of metric_result[.][.] should match shape of Ymean_val."
        )
        
        return metric_results

    def _select_best_kernels(self, metric_result):
        """
        Selects the best-performing kernel for each column across multiple metrics by comparing their
        column-wise mean values.
        The process includes:
            1. **Compute Column-wise Mean**: For each metric and kernel, calculate the mean values column-wise.
            2. **Compare and Update Scores**:
                - For each column, compare the mean values of each kernel pair across all metrics.
                - The kernel with a lower (better) mean value than another kernel gains a point, and the other kernel loses a point.
                - This scoring is done for every column independently, so each kernel's performance is evaluated on a per-column basis.
                - The result is a cumulative score for each kernel in each column, reflecting its overall performance relative to other kernels.
            3. **Select Best Kernels**: Identify the kernel with the highest score for each column.

        Parameters:
            metric_result (dict): A nested dictionary containing the computed metrics for each kernel.
                    - Outer Key (str): The name of the metric (e.g., 'KL Divergence', 'Hellinger Distance', 'Wasserstein Distance').
                    - Outer Value (dict): 
                        - Inner Key (str): The name of the kernel (e.g., 'RBF', 'Matern52').
                        - Inner Value (array-like): The computed metric values for the validation data.
    
        Returns:
            best_kernels (list): List of best-performing kernels for each column.
        """
        # Step 1: Compute the column-wise mean for each metric and kernel
        mean_columnwise = {}
        
        for metric_name in metric_result:
            mean_columnwise[metric_name] = {}
            for kernel_name in metric_result[metric_name]:
                mean_columnwise[metric_name][kernel_name] = np.mean(metric_result[metric_name][kernel_name], axis=0)
        
        num_columns = len(mean_columnwise[metric_name][kernel_name])  # extracting the number of output dimensions
        assert num_columns == self.Y_mean.shape[1], "Error in _select_best_kernels: number of output dimensions must match." 

        
        # Initialize kernel scores to zero
        kernel_scores = {i: {kernel_name: 0 for kernel_name in mean_columnwise[metric_name]} for i in range(num_columns)}
    
        # Step 2: Compare metrics and update scores
        for metric_name in mean_columnwise:
            for i in range(num_columns):  # loop through columns
                for kernel_i in mean_columnwise[metric_name]:
                    for kernel_j in mean_columnwise[metric_name]:
                        if mean_columnwise[metric_name][kernel_i][i] < mean_columnwise[metric_name][kernel_j][i]:
                            kernel_scores[i][kernel_i] += 1
                            kernel_scores[i][kernel_j] -= 1
                        elif mean_columnwise[metric_name][kernel_i][i] > mean_columnwise[metric_name][kernel_j][i]:
                            kernel_scores[i][kernel_i] -= 1
                            kernel_scores[i][kernel_j] += 1
    
        # Step 3: Determine the best kernel for each column
        best_kernels = []
        
        for scores in kernel_scores.values():
            best_kernel = max(scores, key=scores.get)  # Find the kernel with the maximum score
            best_kernels.append(best_kernel)  # Append the best kernel to the list
    
        return best_kernels
########################################################################################################


