import numpy

from setuptools import setup
from Cython.Build import cythonize


# This project nominally configures package management via pyproject.toml.
# However, that file is not setup to handle the Cython builds.  Therefore, we
# presently include this call to ensure that the Cython build does proceed as
# necessary.
setup(
    ext_modules=cythonize("surmise/emulationsupport/matern_covmat.pyx",
                          language_level = "3"),
    include_dirs=[numpy.get_include()]
)
