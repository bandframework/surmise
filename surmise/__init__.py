__version__ = "0.1.1"
__author__ = 'Matthew Plumlee, Özge Sürer, Stefan M. Wild'
__credits__ = 'Northwestern University, Argonne National Laboratory'

import os

f_dir = os.path.dirname(os.path.realpath(__file__))
__calibrationmethods__ = [f for f in os.listdir(f_dir + '/calibrationmethods')
                          if '.py' in f and '__' not in f]
__emulationmethods__ = [f for f in os.listdir(f_dir + '/emulationmethods')
                        if '.py' in f and '__' not in f]
__utilitiesmethods__ = [f for f in os.listdir(f_dir + '/utilitiesmethods')
                        if '.py' in f and '__' not in f]
