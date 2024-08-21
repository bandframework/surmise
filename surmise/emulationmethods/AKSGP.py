#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from joblib import Parallel, delayed
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Emulator:
    def __init__(self, X, Y_mean, Y_std):
        """
        Initializes the Emulator class.

        Parameters:
            X (array-like): Input features of shape (n_samples, n_features).
            Y_mean (array-like): Mean values of the training data of shape (n_samples, n_outputs).
            Y_std (array-like): Standard deviation of the training data of shape (n_samples, n_outputs).
            
        Raises:
            ValueError: 
                - If the arrays `X`, `Y_mean`, and `Y_std` contains NaN, inf, or any non-numeric values.
                - If the number of training points in `X` and `Y_mean` do not match.
                - If the shapes of `Y_mean` and `Y_std` do not match.
        """

        # Validate the arrays
        self._validate_array(X, 'X')
        self._validate_array(Y_mean, 'Y_mean')
        self._validate_array(Y_std, 'Y_std')

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


    def _kernel_list(self):
        """
        List of all available kernels to train the GPs. Add more kernels here to increase options. 
        If "fit (kernel=AKS)", best kernels among the list will be selected for each output dimension.
        
        Parameters:
            input_dim (int): Dimensions of the input space in which the GPs are trained.
            
        Returns:
            dict: A dictionary containing all kernels.
        """
        # accessing the dimensionality of input space
        input_dim = self.X.shape[1]
        
        # kernels that can be used. Add more kernels here to increase options
        kernels = {
        'Matern12': 1.0 * Matern(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(1e-05, 1e20), nu=0.5),
        'Matern32': 1.0 * Matern(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(1e-05, 1e20), nu=1.5),
        'Matern52': 1.0 * Matern(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(1e-05, 1e20), nu=2.5),
        'RBF': 1.0 * RBF(length_scale=np.ones(input_dim)/2.0, length_scale_bounds=(1e-05, 1e20))
        }
        return kernels

    
    def _define_metrics(self):
        """
        Defines a dictionary of metrics to compare two Gaussian distributions.
    
        Returns:
            dict: A dictionary containing functions for the defined metrics.
        """

        # Different metrics to compare distributions and select the best kernels.
        # Define more metrics here that would be considered in the automatic kernel selection process.
        def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
            return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2.0 * sigma2**2) - 0.5
    
        def hellinger_distance_gaussian(mu1, sigma1, mu2, sigma2):
            term1 = np.sqrt(2.0 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))
            term2 = np.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
            return np.sqrt(1.0 - term1 * term2)
    
        def wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2):
            return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)
    
        # Define metrics in a dictionary
        metrics = {
            'KL Divergence': kl_divergence_gaussian,
            'Hellinger Distance': hellinger_distance_gaussian,
            'Wasserstein Distance': wasserstein_distance_gaussian
        }
    
        return metrics

    
    def fit(self, kernel: str = 'AKS', nrestarts: int = 10, n_jobs: int = -1, seed: int = 42) -> None:

        """
        Train individual Gaussian Processes (GPs) for each output dimension.
        
        Parameters:
            kernel (str): The type of kernel to use for the Gaussian Process. Default is 'AKS' -- Automatic kernel selection. 
            List of kernels should be defined in "kernel_list()". Options include:
                - 'AKS' : The function will train GPs with all kernels defined in 'kernel_list' and automatically select the best one.
                          The best kernel is chosen by fitting GPs (with 90% of training data) using each kernel and evaluating their 
                          performance using different distance metrics defined in "_define_metrics()" (like KL divergence, Hellinger dist,
                          Wasserstein dist) on the rest 10% of the traning data. See the "_AutoKernelSelection" function for more details.
                          The GPs are then retrained with the selected kernels for each output dimension with all training data.
                - 'RBF': Radial Basis Function kernel.
                - 'Matern12': Matern kernel with nu=0.5.
                - 'Matern32': Matern kernel with nu=1.5.
                - 'Matern52': Matern kernel with nu=2.5.
            
            nrestarts (int, optional): Number of restarts for the optimizer to improve convergence. Default is 10.
            
            n_jobs (int, optional): Number of CPU cores to use for parallel processing. 
                                    If set to -1, all available cores will be used. Default is -1.

            seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 42.
                          
        Raises:
            Exception: If an error occurs during the GP fitting process.
        """

        # load the list of all available kernels
        kernels = self._kernel_list()
        
        # If kernel='AKS':
        #   - Split training data in 90% - 10% batch. 
        #   - Use 90% (training) batch to fit GPs for all output dimensions with all kernels in "kernel_list".
        #   - Use 10% (pseudo_test) batch to select best kernels for each output dimension.
        #   - Retrain the GPs with the selected kernels for each output dimension with all training data (before split).
        
        if kernel=='AKS':
            logger.info(
                f"Automatic kernel selection opted. Best kernel for each output dimension will be selected from the list of kernels:\n"
                f"   {list(kernels.keys())}\n"
            )

            # Split training data in: 90% (training) - 10% (pseudo_test) batch
            X_train, Ymean_train, Ystd_train, pseudo_test_X, pseudo_test_Ymean, pseudo_test_Ystd = self.split_train_test(
                                                                                                        X=self.X, 
                                                                                                        Ymean=self.Y_mean, 
                                                                                                        Ystd=self.Y_std, 
                                                                                                        train_ratio=0.9, 
                                                                                                        seed=seed)
        
            logger.info(f"Shape of training arrays: {X_train.shape}, {Ymean_train.shape}, {Ystd_train.shape}")
            logger.info(f"Shape of pseudo_test arrays: {pseudo_test_X.shape}, {pseudo_test_Ymean.shape}, {pseudo_test_Ystd.shape}")


            try:
                # Initializing GP dictionary to store the fitted GPs for all available kernels
                gplist = {}

                # Training GPs with different kernels for all output dimensions in parallel
                logger.info("Training GPs with all available kernels...")

                # Standardize data before GP training
                X_stnd, Ymean_stnd, Ystd_stnd = self._preprocess_data(X=X_train, Y_mean=Ymean_train, Y_std=Ystd_train)

                for kernel_name, ker in kernels.items():
                    gplist[kernel_name] = Parallel(n_jobs=n_jobs)(
                        delayed(self.fit_singleGP)(Xfit=X_stnd, 
                                                     Yfit_mean=sample_column, 
                                                     Yfit_std=Ystd_stnd[:, i], 
                                                     kernel=ker, 
                                                     nrestarts=nrestarts
                                                    )
                        for i, sample_column in enumerate(Ymean_stnd.T)
                    )
                    logger.info(f"  Trained GPs with {kernel_name} kernels.")

                # Find best kernel for each output dimension
                logger.info("Finding best kernels for each output dimension...")
                best_kernels = self._AutoKernelSelection(
                                    GP_dict=gplist, 
                                    kernels=kernels, 
                                    test_X=pseudo_test_X, 
                                    test_Ymean=pseudo_test_Ymean, 
                                    test_Ystd=pseudo_test_Ystd)
                
                
                assert len(Ymean_stnd[1]) == len(best_kernels), (
                "Error during GP fit: Number of best kernels not equal to number of output dimension."
                "Check the _AutoKernelSelection function."
                )
                
                del gplist  # Free memory of the gplist objects

                # Retrain the GPs with the selected kernels for each output dimension with all training data in parallel
                # Append the trained GPs to self.gps
                logger.info("Retraining the GPs with selected best kernels using all training data...")

                # Standardize data before GP retraining
                X_stnd, Ymean_stnd, Ystd_stnd = self._preprocess_data(X=self.X, Y_mean=self.Y_mean, Y_std=self.Y_std)

                self.gps = Parallel(n_jobs=n_jobs)(
                    delayed(self.fit_singleGP)(Xfit=X_stnd, 
                                                 Yfit_mean=sample_column, 
                                                 Yfit_std=Ystd_stnd[:, i], 
                                                 kernel=kernels[best_kernels[i]], 
                                                 nrestarts=nrestarts
                                                )
                    for i, sample_column in enumerate(Ymean_stnd.T)
                )
                logger.info("Retraining GPs complete.\n")
            
            except Exception as e:
                logger.error(f"Error during GP fit with automatic kernel selection: {e}")
                raise


        else:
            # Check if the specified kernel is valid and select the specified kernel
            if kernel not in kernels:
                raise ValueError(f"Unsupported kernel type: {kernel}. Available kernels: {list(kernels.keys())}")
            ker = kernels[kernel]
    
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

    
    def predict(self, X_new, return_covariance=False):
        """
        Predicts from fitted GP at new input points.

        Parameters:
            X_new (array-like): predict input points.
            return_covariance (bool): Whether to return the full covariance matrix. Defaults to False.

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

            # Transform the standard deviations or covariances matrix back to the original scale
            scale = self._scaler_Y.scale_
            if return_covariance:
                covariances = [covariance * scale[i]**2 for i, covariance in enumerate(covariances_scaled)]
                return means, covariances
            else:
                std_devs = [np.sqrt(np.diag(covariance)) * scale[i] for i, covariance in enumerate(covariances_scaled)]
                std_devs = np.column_stack(std_devs)
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
        

    def split_train_test(self, X, Ymean, Ystd, train_ratio=0.9, seed=42):
        """
        Splits the dataset into training and test sets based on the specified train_ratio.
    
        Parameters:
            X (array-like): Input features of shape (n_samples, n_features).
            Ymean (array-like): Mean values of the training data of shape (n_samples, n_outputs).
            Ystd (array-like): Standard deviation of the training data of shape (n_samples, n_outputs).
            train_ratio (float, optional): The ratio of the dataset to be used for training. Default is 0.9 (90%).
            seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 42.
    
        Returns:
            X_train (array-like): Training set of input features.
            Ymean_train (array-like): Training set of mean values.
            Ystd_train (array-like): Training set of standard deviations.
            X_test (array-like): Test set of input features.
            Ymean_test (array-like): Test set of mean values.
            Ystd_test (array-like): Test set of standard deviations.
        """
        # Set the seed for reproducibility
        np.random.seed(seed)
    
        # Calculate the size of the training set as a percentage of the total data
        train_size = int(train_ratio * X.shape[0])
        
        # Generate 'train_size' random unique indices
        selected_indices = np.random.choice(X.shape[0], size=train_size, replace=False)
        
        # Select the rows for training
        X_train = X[selected_indices]
        Ymean_train = Ymean[selected_indices]
        Ystd_train = Ystd[selected_indices]
        
        # Select the remaining rows for the test set
        X_test = np.delete(X, selected_indices, axis=0)
        Ymean_test = np.delete(Ymean, selected_indices, axis=0)
        Ystd_test = np.delete(Ystd, selected_indices, axis=0)
    
        return X_train, Ymean_train, Ystd_train, X_test, Ymean_test, Ystd_test
    

    def compare_dist_metrics(self, P_mean, P_std, Q_mean, Q_std):
        """
        Computes different metrics to compare two Gaussian distributions P and Q.
    
        Parameters:
            P_mean (array-like): Mean values of the distribution to compare with the reference distribution Q.
            P_std (array-like): Standard deviation of the distribution to compare with the reference distribution Q.
            Q_mean (array-like): Mean values of the reference distribution Q.
            Q_std (array-like): Standard deviation of the reference distribution Q.
    
        Returns:
            dict: A dictionary containing arrays with the metrics defined in "_define_metrics()" for each element:
                  - 'KL Divergence': Kullback-Leibler divergence.
                  - 'Hellinger Distance': Hellinger distance.
                  - 'Wasserstein Distance': Wasserstein distance.
                  - ...
        """
    
        # If passed as lists, convert to numpy arrays for element-wise operations
        P_mean = np.array(P_mean)
        P_std = np.array(P_std)
        Q_mean = np.array(Q_mean)
        Q_std = np.array(Q_std)
    
        # Ensure that shapes of the arrays match
        assert P_mean.shape == P_std.shape == Q_mean.shape == Q_std.shape, (
            "Error in compare_dist_metrics: All input arrays must have the same shape."
        )
    
        # Retrieve the metrics dictionary
        metrics = self._define_metrics()
    
        # Compute and store metric results in a dictionary
        results = {metric_name: metric_func(P_mean, P_std, Q_mean, Q_std) 
                   for metric_name, metric_func in metrics.items()}
    
        return results

    
    def _AutoKernelSelection(self, GP_dict, kernels, test_X, test_Ymean, test_Ystd):        
        """
        Automatically select the best kernel by comparing the performance of different kernels using various 
        distance metrics between the predicted and actual distributions of the test data.
        
        Parameters:
            GP_dict (dict): Dictionary of fitted GPs for each kernel.
            kernels (dict): Dictionary of kernel options.
            test_X (array-like): Input features of the test data.
            test_Ymean (array-like): Mean values of the test data.
            test_Ystd (array-like): Standard deviations of the test data.
            
        Returns:
            list: The best kernels for each output dimension.
        """
    
        def compute_metric_results():
            """Helper function to compute metric results for each kernel."""
            metric_results = {metric: {} for metric in self.compare_dist_metrics([], [], [], []).keys()}
            for kernel_name in kernels.keys():
                self.gps = GP_dict[kernel_name]
                GP_means, GP_stds = self.predict(test_X, return_covariance=False)
                results = self.compare_dist_metrics(P_mean=GP_means, P_std=GP_stds, Q_mean=test_Ymean, Q_std=test_Ystd)
                for metric_name, value in results.items():
                    metric_results[metric_name][kernel_name] = value
            return metric_results
    
        def compute_compare_dict(metric_results):
            """Helper function to compute percentage comparisons between kernels."""
            compare = {}
            for metric_name, metric_result in metric_results.items():
                compare[metric_name] = {}
                for i, kernel_i in enumerate(kernels.keys()):
                    for j, kernel_j in enumerate(kernels.keys()):
                        if i < j:
                            if kernel_i not in compare[metric_name]:
                                compare[metric_name][kernel_i] = {}
                            diff = metric_result[kernel_i] - metric_result[kernel_j]
                            positive_percentage = np.mean(diff > 0, axis=0) * 100               
                            compare[metric_name][kernel_i][kernel_j] = positive_percentage
            return compare
    
        def compute_averaged_comparison(compare):
            """Helper function to average the comparisons across metrics."""
            averaged_comparison = {}
            for kernel_i in kernels.keys():
                averaged_comparison[kernel_i] = {}
                for kernel_j in kernels.keys():
                    if kernel_i < kernel_j:
                        percentages = [compare[metric_name][kernel_i][kernel_j] 
                                       for metric_name in metric_results.keys() 
                                       if kernel_j in compare[metric_name][kernel_i]]
                        if percentages:
                            averaged_comparison[kernel_i][kernel_j] = np.mean(percentages, axis=0)
            return averaged_comparison
    
        def select_best_kernels(averaged_comparison):
            """Helper function to select the best kernel for each column."""
            num_columns = test_Ymean.shape[1]
            best_kernels = []
            for col in range(num_columns):
                kernel_scores = {kernel: 0 for kernel in kernels.keys()}
                for kernel_i, comparisons in averaged_comparison.items():
                    for kernel_j, percentage in comparisons.items():
                        if percentage[col] < 50:
                            kernel_scores[kernel_i] += 1
                        else:
                            kernel_scores[kernel_j] += 1
                best_kernel = max(kernel_scores, key=kernel_scores.get)
                best_kernels.append(best_kernel)
            return best_kernels

        
        # Step 1: Compute metric results
        metric_results = compute_metric_results()
    
        # Step 2: Compute compare dictionary
        compare = compute_compare_dict(metric_results)
    
        # Step 3: Average the comparisons across metrics
        averaged_comparison = compute_averaged_comparison(compare)
    
        # Step 4: Select the best kernels
        best_kernels = select_best_kernels(averaged_comparison)

        logger.info(f"Selected best kernels for each output dimension:\n   {best_kernels}\n")
        
        return best_kernels
                


###############################################################
