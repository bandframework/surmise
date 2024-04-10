#!/bin/bash

python -m pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
pip install --upgrade tox
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
