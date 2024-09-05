#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import dill
import gzip

from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).resolve().parent  # full path to this file's directory

from metrics import rmse, normalized_rmse, dss, intervalstats
from metrics import kl_divergence_gaussian, hellinger_distance_gaussian, wasserstein_distance_gaussian


import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

#----------------------------------------------------------------

def kfold_test_emulators(path_kfold_data, kfold_seeds_filename, methods):
    """Process each k-fold set and compute metrics."""

    k_seeds = load_seeds(kfold_seeds_filename)
    
    for i in range(len(k_seeds)):
        # Load emulators and test data
        emulators = load_emulators(path_kfold_data[i], methods)
        
        # Define the path to the test directory and load data
        test_dir = Path(path_kfold_data[i]) / 'test'
        X_test, Ymean_test, Ystd_test = load_test_data(test_dir)

        # Standardize the test data
        Ymean_test, Ystd_test, scaler_Y = standardize_data(Ymean_test, Ystd_test)
        Yvar_test = np.square(Ystd_test)
        xloc = np.arange(Ymean_test.shape[1])

        # Initialize a list to store the results
        results = []

        for method, emu in emulators.items():
            predmean, predvar = compute_predictions(emu, method, X_test, xloc, scaler_Y)
            EC, RMSE, NRMSE, KLdiv, HD, WD = compute_metrics(predmean, predvar, Ymean_test, Yvar_test)

            # Store the results in the list
            results.append({
                'Method': method,
                'Empirical Coverage': ', '.join(['{:.6f}'.format(val) for val in EC]),
                'RMSE': '{:.6f}'.format(RMSE),
                'NRMSE': '{:.6f}'.format(NRMSE),
                'KL Divergence': '{:.6f}'.format(np.mean(KLdiv)),
                'Hellinger Distance': '{:.6f}'.format(np.mean(HD)),
                'Wasserstein Distance': '{:.6f}'.format(np.mean(WD))
            })

        # Convert the list of results into a DataFrame
        results_df = pd.DataFrame(results)
        print_metrics_table(results_df, i)

#----------------------------------------------------------------

def load_seeds(kfold_seeds_filename):
    """Load k-fold seeds from a file."""
    try:
        kfold_seeds_path =  current_dir / kfold_seeds_filename
        with open(kfold_seeds_path, 'r') as file:
            return [int(line.strip()) for line in file]
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error reading seed file '{kfold_seeds_filename}': {e}")
        raise

#----------------------------------------------------------------

def load_emulators(path, methods):
    """Load emulators from dill.gz files."""
    emulators = {}
    path = Path(path)  # Convert to Path object for consistency
    
    for method in methods:
        filename = path / f'emulator_{method}.dill.gz'
        
        if not filename.exists(): 
            logging.warning(f"File not found: {filename}")
            continue
        
        try:
            with gzip.open(filename, 'rb') as f:
                emulators[method] = dill.load(f)
        except Exception as e:
            logging.error(f"Error loading emulator from {filename}: {e}")
    
    return emulators

#----------------------------------------------------------------

def load_test_data(test_dir):
    """Load test data from text files."""
    test_dir = Path(test_dir)  # Convert to Path object for consistency

    X_test = np.loadtxt(test_dir / 'X.txt')
    Ymean_test = np.loadtxt(test_dir / 'Ymean.txt')
    Ystd_test = np.loadtxt(test_dir / 'Ystd.txt')
    
    return X_test, Ymean_test, Ystd_test
    
#----------------------------------------------------------------

def standardize_data(Ymean_test, Ystd_test):
    """Standardize the test data."""
    scaler_Y = StandardScaler()
    Ymean_test = scaler_Y.fit_transform(Ymean_test)
    Ystd_test = Ystd_test / scaler_Y.scale_
    return Ymean_test, Ystd_test, scaler_Y
    
#----------------------------------------------------------------
    
def compute_predictions(emu, method, X_test, xloc, scaler_Y):
    """Compute predictions using the emulator."""
    if method in {'PCGP', 'PCSK'}:
        pred = emu.predict(x=xloc, theta=X_test)
        predmean = pred.mean().T
        predvar = pred.var().T
    elif method in {'AKSGP', 'PCGP_scikit'}:
        predmean, predstd = emu.predict(X_test)
        predvar = np.square(predstd)
    
    predmean = scaler_Y.transform(predmean)
    predvar = np.square(np.sqrt(predvar) / scaler_Y.scale_)
    return predmean, predvar

#----------------------------------------------------------------


def compute_metrics(predmean, predvar, Ymean_test, Yvar_test):
    """Compute various metrics for predictions."""
    EC = intervalstats(Ymean_test, predmean, predvar)
    RMSE = rmse(Ymean_test, predmean)
    NRMSE = normalized_rmse(Ymean_test, predmean)

    # Calculate the KL divergence, Hellinger distance, and Wasserstein distance using the diagonal covariance matrices
    
    # Initialize array's to store the distances
    kl_div = np.zeros(predmean.shape)
    wasserstein_dist = np.zeros(predmean.shape)
    hellinger_dist = np.zeros(predmean.shape)
    
    # Loop over each pair of means and variances
    for i in range(predmean.shape[0]):
        for j in range(predmean.shape[1]):
            mu1 = predmean[i, j]
            mu2 = Ymean_test[i, j]
            var1_ij = predvar[i, j]
            var2_ij = Yvar_test[i, j]
            
            # Calculate the distances for the current pair
            kl_div[i, j] = kl_divergence_gaussian(mu1=mu1, Cov1=var1_ij, mu2=mu2, Cov2=var2_ij)
            hellinger_dist[i, j] = hellinger_distance_gaussian(mu1=mu1, Cov1=var1_ij, mu2=mu2, Cov2=var2_ij)
            wasserstein_dist[i, j] = wasserstein_distance_gaussian(mu1=mu1, Cov1=var1_ij, mu2=mu2, Cov2=var2_ij)

    return EC, RMSE, NRMSE, kl_div, hellinger_dist, wasserstein_dist

#----------------------------------------------------------------

def print_metrics_table(results_df, kfold_set_index):
    """Print the metrics in a formatted table."""
    logger.info(f"Metrics for k-fold set {kfold_set_index} >>>>>>\n")
    
    # Log the header directly
    header = f"{'Method':<12} {'Empirical Coverage':<27} {'RMSE':<14} {'NRMSE':<12} {'KL Divergence':<15} {'Hellinger Distance':<20} {'Wasserstein Distance':<20}"
    logger.info(header)
    logger.info('-' * len(header))
    
    # Log each row in the DataFrame
    for index, row in results_df.iterrows():
        logger.info(f"{row['Method']:<12} {row['Empirical Coverage']:<25} {row['RMSE']:<15} {row['NRMSE']:<15} {row['KL Divergence']:<17} {row['Hellinger Distance']:<20} {row['Wasserstein Distance']:<20}")

    logger.info("\n")


