import setuptools
from setuptools import setup
from setuptools.command.test import test as TestCommand


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
    version="0.1",
    author="Matthew Plumlee, Özge Sürer, Stefan M. Wild",
    author_email="ozgesurer2019@u.northwestern.edu",
    description="Surrogate model interface for calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/surmising/surmise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
                      'numpy',
                      'scipy'
                      ],
    cmdclass={'test': Run_TestSuite}
)