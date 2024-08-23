#!/usr/bin/env python3
import sys
import os
import numpy as np
import dill
import gzip
import logging

sys.path.append(os.path.abspath('../../surmise/emulationmethods'))
from AKSGP import Emulator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('emulator_train_allKernels.log', mode='w'),  # Log to file
        logging.FileHandler('emulator_train_baseKernels.log', mode='w'),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)


# Load training data
train_dir = 'simulation_data/Grad_Pb-Pb-2760GeV/train'
X = np.loadtxt(os.path.join(train_dir, 'X.txt'))
Ymean = np.loadtxt(os.path.join(train_dir, 'Ymean.txt'))
Ystd = np.loadtxt(os.path.join(train_dir, 'Ystd.txt'))

logger.info(f"Arrays loaded from directory '{train_dir}'.")
logger.info(f"Shapes of loaded arrays: {X.shape}, {Ymean.shape}, {Ystd.shape}")


# Load testing data
test_dir = 'simulation_data/Grad_Pb-Pb-2760GeV/test'

Xval = np.loadtxt(os.path.join(test_dir, 'X.txt'))
Ymeanval = np.loadtxt(os.path.join(test_dir, 'Ymean.txt'))
Ystdval = np.loadtxt(os.path.join(test_dir, 'Ystd.txt'))

logger.info(f"Arrays loaded from directory '{test_dir}'.")
logger.info(f"Shapes of loaded arrays: {Xval.shape}, {Ymeanval.shape}, {Ystdval.shape}")


# Emulator training -------->

# Train GPs
emu = Emulator(X=X_train, Y_mean=Ymean_train, Y_std=Ystd_train)
emu.fit(kernel='AKS', nrestarts=20)


# Save the trained emulator object to a file using dill and gzip

# with gzip.open('emulator_AKS_allKernels.dill.gz', 'wb') as f:
with gzip.open('emulator_AKS_baseKernels.dill.gz', 'wb') as f:
    dill.dump(emu, f)






