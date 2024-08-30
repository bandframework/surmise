#!/usr/bin/env python3

import sys, os
import shutil
import numpy as np
import dill
import gzip
import logging

sys.path.append(os.path.abspath('../../..'))  # surmise directory
from sklearn.model_selection import train_test_split
from surmise.emulation import emulator
from surmise.emulationmethods.AKSGP import Emulator as emulator_AKSGP


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def kfold_train_save_emulators(simulation_name, 
                               methods = ['PCGP', 'PCSK', 'AKSGP'],
                               kfold_seeds_filename='kfold_seeds.txt', 
                               subdir_name='kfold_set', 
                               regenerate=False, 
                               retrain=False):
    """
    Main function to generate k-fold datasets and train and save Gaussian Process emulators.

    Parameters:
    -----------
    simulation_name : str
        The name of the simulation, used to locate simulation data and name directories.
    kfold_seeds_filename : str
        The filename of the seed file used for generating k-fold splits.
    methods : list
        Methods for emulator training
    subdir_name : str, optional
        Sub directories where the trained emulator files will be saved.
    regenerate : bool, optional
        If True, delete k-fold directories, including all saved emulators, and regenerate datasets (default: False).
    retrain : bool, optional
        If True, retrain emulators even if they already exist (default: False).

    Returns:
    --------
    None
    """
    # Load simulation data
    SimData_dir = f'../simulation_data/{simulation_name}'
    X = np.loadtxt(os.path.join(SimData_dir, 'X.txt'))
    Ymean = np.loadtxt(os.path.join(SimData_dir, 'Ymean.txt'))
    Ystd = np.loadtxt(os.path.join(SimData_dir, 'Ystd.txt'))

    # Slice the data (optional)
    X = X[:100, :]
    Ymean = Ymean[:100, :2]
    Ystd = Ystd[:100, :2]

    # Generate k-fold datasets
    basedir_name = f'{simulation_name}_datasets'
    path_kfold_data = generate_kfold_sets(
        X=X,
        Y_mean=Ymean,
        Y_std=Ystd,
        base_dir=basedir_name,
        test_size=0.2,
        regenerate=regenerate,
        seed_file=kfold_seeds_filename,
        sub_dir=subdir_name
    )

    # Train and save emulators for each k-fold dataset
    for kfold_dir in path_kfold_data:
        train_dir = os.path.join(kfold_dir, 'train')
        logger.info(f"Training and saving emulators on k-fold training data in '{train_dir}'>>>>\n")
        
        # Load the training data
        X_train = np.loadtxt(os.path.join(train_dir, 'X.txt'))
        Ymean_train = np.loadtxt(os.path.join(train_dir, 'Ymean.txt'))
        Ystd_train = np.loadtxt(os.path.join(train_dir, 'Ystd.txt'))
        
        # Train and save emulators
        train_save_emu(X_train, Ymean_train, Ystd_train, methods, kfold_dir, retrain=retrain)

    logger.info("All k-fold datasets processed and all emulators trained.\n")

#----------------------------------------------------------------

def train_save_emu(X, Ymean, Ystd, methods, emulator_save_dir, retrain=False):
    """
    Train and save Gaussian Process emulators using specified methods.

    Parameters:
    -----------
    X : np.ndarray
        The input parameter space (theta) for training the emulator.
    Ymean : np.ndarray
        The mean response values corresponding to the input parameters.
    Ystd : np.ndarray
        The standard deviation of the response values corresponding to the input parameters.
    methods : list
        A list of methods to use for training the emulators. Options include 'PCGP', 'PCSK', and 'AKSGP'.
    emulator_save_dir : str
        Directory where the trained emulator files will be saved.
    retrain : bool, optional
        If True, retrain the emulator even if it already exists (default: False).

    Returns:
    --------
    None
        Saves the trained emulators in the specified directory.
    """
    
    xloc = np.arange(Ymean.shape[1])  # refers to the observable indices

    emus = {}
    for method in methods:
        
        # Check if the emulator file already exists
        filename = os.path.join(emulator_save_dir, f'emulator_{method}.dill.gz')
        if os.path.exists(filename) and not retrain:
            logger.info(f"Emulator '{method}' already exists at '{filename}'. Skipping training. Set retrain=True to retrain emulator.\n")
            continue
        
        if method == 'PCGP':
            prior = {'min': np.min(Ymean.T), 'max': np.max(Ymean.T)}
            args = {'prior': prior}
            emus[method] = emulator(x=xloc, theta=X, f=Ymean.T, method=method, args=args)
        elif method == 'PCSK':
            args = {'simsd': Ystd.T}
            emus[method] = emulator(x=xloc, theta=X, f=Ymean.T, method=method, args=args)
        elif method == 'AKSGP':
            emus[method] = emulator_AKSGP(X, Y_mean=Ymean, Y_std=Ystd)
            emus[method].fit(kernel='AKS', nrestarts=10, seed=None)
        else:
            logger.error(f"Unknown method '{method}'.\n")
            continue
            
        # Saving the emulators after training
        try:
            with gzip.open(filename, 'wb') as f:
                dill.dump(emus[method], f)
            logger.info(f"Emulator '{method}' trained and saved at '{filename}'\n")
        except Exception as e:
            logger.error(f"Failed to save emulator '{method}': {e}\n")


#----------------------------------------------------------------

def generate_kfold_sets(
    X: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    base_dir: str,
    test_size: float = 0.2,
    regenerate: bool = False,
    seed_file: str = 'kfold_seeds.txt',
    sub_dir: str = 'kfold_set'
) -> list:
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
    base_dir : str
        Base directory name for saving datasets.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default: 0.2).
    regenerate : bool, optional
        If True, regenerate datasets; if False, skip if directories exist (default: False).
    seed_file : str, optional
        Path to the file containing random seeds, one per line (default: 'kfold_seeds.txt').
    sub_dir : str, optional
        Base name for subdirectories within the main directory, formatted as '{sub_dir}i', 
        where i ranges from 1 to the number of seeds in seed_file (default: 'kfold_set').

    Returns:
    --------
        str: Name of base directory where data are saved.
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
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error reading seed file '{seed_file}': {e}")
        raise

    # Check if base directory exists
    if os.path.exists(base_dir):
        if regenerate:
            # If regenerate is True, delete and recreate the directory
            shutil.rmtree(base_dir)
            os.makedirs(base_dir)
        else:
            # If regenerate is False, skip the rest of the code
            logger.info(
                f"'{base_dir}' already exists. Skipping k-fold dataset generation.\n"
                f"To regenerate datasets, set regenerate=True. WARNING: This will remove all contents, "
                f"including saved emulators, inside '{base_dir}'.\n"
            )
            # Return existing k-fold directories
            return [os.path.join(base_dir, f'{sub_dir}{i}') for i in range(1, len(k_seeds) + 1)]
            
            # # List to store paths of k-fold directories
            # kfold_dirs = []
            # for i, seed in enumerate(k_seeds, start=1):
            #     kfold_set_dir = os.path.join(base_dir, f'{sub_dir}{i}')
            #     kfold_dirs.append(str(kfold_set_dir))

            # return kfold_dirs
    else:
        # If the directory does not exist, create it
        os.makedirs(base_dir)

    # List to store paths of k-fold directories
    kfold_dirs = []

    # Create subdirectories for the k-fold sets: train and test
    for i, seed in enumerate(k_seeds, start=1):
        kfold_set_dir = os.path.join(base_dir, f'{sub_dir}{i}')
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

        # Add kfold_set_dir to the list
        kfold_dirs.append(str(kfold_set_dir))

    logger.info(f"Datasets saved in '{base_dir}'.\n")

    return kfold_dirs

