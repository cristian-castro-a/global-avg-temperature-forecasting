#!/bin/bash

conda create --name py39 python=3.9

# Tensorflow in Mac M1
conda install -c apple tensorflow-deps -y
python -m pip install tensorflow-macos
pip install tensorflow-metal

# Other
conda install -c anaconda pandas
conda install -c conda-forge hydra-core
conda install -c plotly plotly
# Up to here numpy is version 1.22.3 - In the past it worked with version 1.23.5

# Update numpy to avoid: "RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf"
pip install numpy --upgrade
conda install -c conda-forge matplotlib
pip install numpy==1.23.5
conda install -c anaconda scikit-learn