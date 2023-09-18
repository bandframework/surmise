import setuptools
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


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="surmise",
    # version="0.1.1",
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    setup_requires=[
        "setuptools>=50.0",
        "setuptools_scm[toml]>=6.0",
        "numpy>=1.18.3",
        "cython",
        "wheel"
    ],
    install_requires=[
                      'numpy>=1.18.3',
                      'scipy>=1.7'
                      ],
    extras_require={'docs': ['sphinx', 'sphinxcontrib.bibtex', 'sphinx_rtd_theme']},
    cmdclass={'test': Run_TestSuite},
    ext_modules=[
        Extension('surmise.emulationsupport.matern_covmat',
                  sources=['surmise/emulationsupport/matern_covmat.pyx']),
    ],
    include_dirs=[numpy.get_include()]
)
