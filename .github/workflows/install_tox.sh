#!/bin/bash

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install --upgrade wheel
python -m pip install --upgrade tox
echo
which python
which pip
which tox
echo
python --version
pip --version
tox --version
echo
pip list
