#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo
    echo "Please pass location of surmise test folder"
    echo
    exit 1
fi
test_path=$1

pushd $test_path                                                   || exit 1
python -m pytest -k "not new"                                      || exit 1
python -m pytest -k "test_new_emu"                                 || exit 1
python -m pytest -k "test_new_cal" --cmdopt2=directbayes           || exit 1
python -m pytest -k "test_new_cal" --cmdopt2=directbayeswoodbury   || exit 1
popd
