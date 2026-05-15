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
tox --version
echo
python -m pip list
echo
