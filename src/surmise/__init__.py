import os

from importlib.metadata import version, PackageNotFoundError

from .test import test

try:
    __version__ = version("surmise")
except PackageNotFoundError:
    # package is not installed
    pass

__author__ = 'Matthew Plumlee, Özge Sürer, Stefan M. Wild, Moses Y-H. Chan'
__credits__ = 'Northwestern University, Argonne National Laboratory'

# General-use public interface
from .emulation import emulator
from .calibration import calibrator

# Advanced API
from .create_sampler import create_sampler

f_dir = os.path.dirname(os.path.realpath(__file__))
__calibrationmethods__ = [f for f in os.listdir(f_dir + '/calibrationmethods')
                          if '.py' in f and '__' not in f]
__emulationmethods__ = [f for f in os.listdir(f_dir + '/emulationmethods')
                        if '.py' in f and '__' not in f]
