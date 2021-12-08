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
    version="0.1.1",
    author="Matthew Plumlee, Özge Sürer, Stefan M. Wild",
    author_email="ozgesurer2019@u.northwestern.edu",
    description="A modular interface for surrogate models and tools",
    url="https://github.com/surmising/surmise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    setup_requires=[
        'setuptools>=18.0',
        'cython'
    ],
    install_requires=[
                      'numpy',
                      'scipy'
                      ],
    extras_require={'extras': ['GPy'],
                    'docs': ['sphinx', 'sphinxcontrib.bibtex', 'sphinx_rtd_theme']},
    cmdclass={'test': Run_TestSuite},
    ext_modules=[
        Extension('surmise.emulationsupport.matern_covmat', sources=['surmise/emulationsupport/matern_covmat.pyx']),
    ],
    include_dirs=[numpy.get_include()]
)
