#!/usr/bin/env python3

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_kfold_sets(
    X: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    exp_name: str,
    test_size: float = 0.2,
    retrain: bool = False,
    seed_file: str = 'kfold_seeds.txt',
    subdir_name: str = 'kfold_set'
) -> None:
    """
    Generate and save k-fold validation datasets by splitting X, Y_mean, and Y_std into train and test sets.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (reshaped to 2D if 1D).
    Y_mean : np.ndarray
        Target mean values (reshaped to 2D if 1D).
    Y_std : np.ndarray
        Target standard deviation values (reshaped to 2D if 1D).
    exp_name : str
        f'{exp_name}_datasets' -> Base directory name for saving datasets.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default: 0.2).
    retrain : bool, optional
        If True, regenerate datasets; if False, skip if directories exist (default: False).
    seed_file : str, optional
        Path to the file containing random seeds, one per line (default: 'kfold_seeds.txt').
    subdir_name : str, optional
        Base name for subdirectories within the main directory, formatted as '{subdir_name}i', 
        where i ranges from 1 to the number of seeds in seed_file (default: 'kfold_set').

    Returns:
    --------
    None
        Saves datasets in 'kfold_set' directories with 'train' and 'test' subdirectories.
    """
    
    # If X, Y_mean, or Y_std are 1D, reshape them to be 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y_mean.ndim == 1:
        Y_mean = Y_mean.reshape(-1, 1)
    if Y_std.ndim == 1:
        Y_std = Y_std.reshape(-1, 1)

    # Read seeds from the text file
    try:
        with open(seed_file, 'r') as file:
            k_seeds = [int(line.strip()) for line in file]
    except FileNotFoundError:
        logger.error(f"Seed file '{seed_file}' not found. Please provide a valid file.")
        return
    except ValueError:
        logger.error(f"Seed file '{seed_file}' contains invalid data. Ensure all lines contain valid integers.")
        return

    
    base_dir = f'{exp_name}_datasets'
    # Check if base directory exists
    if os.path.exists(base_dir):
        if retrain:
            # If retrain is True, delete and recreate the directory
            shutil.rmtree(base_dir)
            os.makedirs(base_dir)
        else:
            # If retrain is False, skip the rest of the code
            logger.info(
                f" '{base_dir}' already exists. Skipping k-fold dataset generation. "
                f"Set retrain=True to regenerate datasets and retrain the emulators.\n"
            )
            return
    else:
        # If the directory does not exist, create it
        os.makedirs(base_dir)

    # Create subdirectories for the k-fold sets: train and test
    for i, seed in enumerate(k_seeds, start=1):
        kfold_set_dir = os.path.join(base_dir, f'{subdir_name}{i}')
        train_dir = os.path.join(kfold_set_dir, 'train')
        test_dir = os.path.join(kfold_set_dir, 'test')
        
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        # Split the data
        X_train, X_test, ymean_train, ymean_test, ystd_train, ystd_test = train_test_split(
            X, Y_mean, Y_std, test_size=test_size, random_state=seed
        )

        # Save the arrays in the respective directories
        np.savetxt(os.path.join(train_dir, 'X.txt'), X_train, delimiter=' ', fmt='%.6f')
        np.savetxt(os.path.join(train_dir, 'Ymean.txt'), ymean_train, delimiter=' ', fmt='%.6f')
        np.savetxt(os.path.join(train_dir, 'Ystd.txt'), ystd_train, delimiter=' ', fmt='%.6f')

        np.savetxt(os.path.join(test_dir, 'X.txt'), X_test, delimiter=' ', fmt='%.6f')
        np.savetxt(os.path.join(test_dir, 'Ymean.txt'), ymean_test, delimiter=' ', fmt='%.6f')
        np.savetxt(os.path.join(test_dir, 'Ystd.txt'), ystd_test, delimiter=' ', fmt='%.6f')

    logger.info(f" Datasets saved in '{base_dir}'.\n")

