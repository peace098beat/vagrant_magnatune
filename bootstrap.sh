#!/bin/bash
##################################
# Deep Learning Vagrant Machine  #
##################################

# apt update
apt-get -y update
apt-get -y upgrade

sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y python3-numpy python3-scipy

sudo pip3 install --upgrade pip
sudo pip3 install tensorflow
sudo pip3 install keras


# Miscellaneous
# pip3 install pyyaml cython
# apt-get install -y libhdf5-7 libhdf5-dev
# pip3 install h5py
# pip3 install ipython
# pip3 install jupyter
# apt-get install -y matplotlib

# apt update
apt-get -y update
apt-get -y upgrade

echo 'All set!'