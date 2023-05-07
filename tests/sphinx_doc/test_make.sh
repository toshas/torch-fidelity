#!/usr/bin/env bash
set -e
set -x

ROOT_DIR=$(realpath $(dirname "$0")/../../)

python3 -m venv venv_sphinx
. venv_sphinx/bin/activate

pip3 install -U pip setuptools
pip3 install -r "${ROOT_DIR}/doc/sphinx/requirements.txt"
pip3 install -e ${ROOT_DIR}

cd "${ROOT_DIR}/doc/sphinx"
make html
