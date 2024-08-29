from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand
import numpy


class Run_TestSuite(TestCommand):
    def run_tests(self):
        import os
        import sys
        py_version = sys.version_info[0]
        print('Python version from setup.py is', py_version)
        run_string = "tests/run-tests.sh -p " + str(py_version)
        os.system(run_string)


setup(
    cmdclass={'test': Run_TestSuite},
    ext_modules=[
        Extension('surmise.emulationsupport.matern_covmat',
                  sources=['surmise/emulationsupport/matern_covmat.pyx']),
    ],
    include_dirs=[numpy.get_include()]
)
