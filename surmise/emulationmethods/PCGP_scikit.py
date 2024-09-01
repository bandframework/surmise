#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from joblib import Parallel, delayed
import logging

# Configure logging for the Emulator class
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Emulator:

    def __init__(self, X, Y, npc):
        """
        A class to perform Gaussian Process Regression (GPR) on data with 
        dimensionality reduction using Principal Component Analysis (PCA).
        The class standardizes the input data, reduces dimensionality via PCA, and fits 
        a Gaussian Process to each of the principal components using scikit-learn library.
    
        Inputs:
            X (array-like): Input features of shape (n_samples, n_features).
            Y (array-like): Values of the training data of shape (n_samples, n_outputs).
            npc (int): Number of principal components to retain.
    
        Attributes:
            scaler (StandardScaler): Scaler used for standardizing features.
            pca (PCA): PCA object for dimensionality reduction.
            gps (list): List of Gaussian Process Regressors for each principal component.
            _trans_matrix (np.ndarray): Transformation matrix to inverse transform PCA results.
        """
        
        self.X = self._ensure_2d(X)
        self.Y = self._ensure_2d(Y)
        self.npc = npc

        self.scaler = StandardScaler()
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')

        self.gps = []
        self._trans_matrix = None

    @staticmethod
    def _ensure_2d(array):
        """
        Ensures that the input array is 2D. If it's 1D, reshapes it.
        """
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array

    def _compute_transformation_matrix(self):
        """
        Computes the transformation matrix used to inverse transform PCA results.

        Returns:
            np.ndarray: The transformation matrix combining PCA and scaling effects.
        """
        return (
            self.pca.components_
            * np.sqrt(self.pca.explained_variance_[:, np.newaxis])
            * self.scaler.scale_
        )

    def _preprocess_data(self, Y):
        """
        Preprocesses the target data Y by scaling and applying PCA.
        Retains only the first `self.npc` principal components.

        Args:
            Y (np.ndarray): The target values.

        Returns:
            np.ndarray: Transformed data in the reduced dimensionality space.
        """
        scaled_Y = self.scaler.fit_transform(Y)
        Z = self.pca.fit_transform(scaled_Y)
        return Z[:, :self.npc]

    def fit(self, nrestarts, n_jobs = -1):
        """
        Fits Gaussian Process models to the PCA-transformed target data.

        Args:
            nrestarts (int): Number of restarts for optimizer in GP fitting.
            n_jobs (int, optional): Number of CPU cores to use for parallel processing. 
                                    If set to -1, all available cores will be used. Default is -1.
        """
        Z = self._preprocess_data(self.Y)
        self._trans_matrix = self._compute_transformation_matrix()

        X_min, X_max = np.min(self.X, axis=0), np.max(self.X, axis=0)
        ptp = X_max - X_min

        # Define the kernel combining RBF and WhiteKernel. 
        
        # This is the same kernel and training as was done by Derek (apart from training emulator 
        # on temperature grid for eta/s and zeta/s and not parameters of these quantities) in 
        # https://github.com/derekeverett/js-sims-bayes/blob/master/src/emulator.py
        rbf_kern = 1. * kernels.RBF(length_scale=ptp, length_scale_bounds=np.outer(ptp, (4e-1, 1e2)))
        hom_white_kern = kernels.WhiteKernel(noise_level=.1, noise_level_bounds=(1e-2, 1e2))
        kernel = rbf_kern + hom_white_kern

    
        def fit_gp_for_component(z):
            """Fits a GP model for a single principal component."""
            gp = GPR(
                kernel=kernel,
                alpha=0.1,
                n_restarts_optimizer=nrestarts,
                copy_X_train=False
            )
            return gp.fit(self.X, z)

        # Parallel fitting of GPs to each principal component
        self.gps = Parallel(n_jobs=n_jobs)(
            delayed(fit_gp_for_component)(sample_column) for sample_column in Z.T
        )

        # Log the GP score for each principal component
        for n, (z, gp) in enumerate(zip(Z.T, self.gps)):
            logger.info(f"GP {str(n)} score : {str(gp.score(self.X, z))}")


    def _inverse_transform_mean(self, Z):
        """
        Inverse transforms the PCA-reduced mean values back to the original space.

        Args:
            Z (np.ndarray): PCA-reduced data.

        Returns:
            np.ndarray: Data transformed back to the original space.
        """
        return Z @ self._trans_matrix[:Z.shape[-1]] + self.scaler.mean_

    def _inverse_transform_std(self, std_Z):
        """
        Inverse transforms the standard deviation from PCA space back to the original space.

        Args:
            std_Z (np.ndarray): Standard deviation in PCA-reduced space.

        Returns:
            np.ndarray: Standard deviation transformed back to the original space.
        """
        return std_Z @ np.abs(self._trans_matrix[:std_Z.shape[-1]])

    def predict(self, X_test):
        """
        Predicts the target values and uncertainties for new input data X_test.

        Args:
            X_test (np.ndarray): The input data for prediction.

        Returns:
            tuple: A tuple containing:
                - mean (np.ndarray): Predicted mean values transformed back to original space.
                - gp_std (np.ndarray): Standard deviation of the predictions, transformed back to original space.
        """
        X_test = self._ensure_2d(X_test)
        predictions = [gp.predict(X_test, return_cov=True) for gp in self.gps]
        gp_mean, gp_cov = zip(*predictions)

        # Inverse transform the mean predictions
        mean = self._inverse_transform_mean(np.column_stack(gp_mean))

        # Compute and inverse transform the standard deviation
        gp_var = np.column_stack([c.diagonal() for c in gp_cov])
        gp_std = self._inverse_transform_std(np.sqrt(gp_var))

        return mean, gp_std
