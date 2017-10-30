#!/bin/bash
set -e

source activate py2
ipython kernel install
source activate py3
ipython kernel install

jupyter notebook
