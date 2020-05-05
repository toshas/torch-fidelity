#!/usr/bin/env bash

ROOT=$(realpath $(dirname $0)/..)
cd ${ROOT}
python3 -m pip install --upgrade setuptools wheel twine
python3 setup.py sdist bdist_wheel
echo "Inspect dist and upload with \"python3 -m twine upload dist/*\""
