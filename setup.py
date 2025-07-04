# IMPORTANT
# * Please adjust GitHub actions to match changes to Python version here.
# * Please make sure that all dependence/version changes made here are reflected
#   in the oldest tox task in tox.ini
#
# The version is being set in _version.py by setuptools_scm (see
# pyproject.toml).  No need to handle version manually here.

import numpy
import codecs

from pathlib import Path
from setuptools import (
    setup, find_packages, Extension
)

_PKG_ROOT = Path(__file__).resolve().parent


def readme_rst():
    fname = _PKG_ROOT.joinpath("README.rst")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()

python_requires = ">=3.9"
# numpy & scipy upper limits required by macos-13 GH action tests.  Without
# these, there are segmentation faults.
code_requires = [
    'numpy>=1.22.0,<2.2.0',
    'scipy>=1.9.0,<1.15.0',
    'scikit-learn>=1.2.0',
    'dill>=0.3.8'
]
test_requires = ["pytest"]
extras_require = {}
install_requires = code_requires + test_requires

extensions = [
    Extension("surmise.emulationsupport.matern_covmat",
              ["src/surmise/emulationsupport/matern_covmat.c"],
              include_dirs=[numpy.get_include()])
]

package_data = {
    "surmise": ["emulationsupport/matern_covmat.pyx",
                "emulationsupport/matern_covmat.c"]
}

project_urls = {
    "Source": "https://github.com/bandframework/surmise",
    "Documentation": "https://surmise.readthedocs.io",
    "Tracker": "https://github.com/bandframework/surmise/issues"
}

setup(
    name="surmise",
    author="Matthew Plumlee, Özge Sürer, Stefan M. Wild, and Moses Y.-H. Chan",
    author_email="moses.chan@northwestern.edu",
    maintainer="Moses Y.-H. Chan",
    maintainer_email="moses.chan@northwestern.edu",
    description="A modular interface for surrogate models and tools",
    long_description=readme_rst(),
    long_description_content_type="text/rst",
    url="https://github.com/bandframework/surmise",
    project_urls=project_urls,
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=package_data,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    ext_modules=extensions,
    keywords="surmise",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
