#!bin/bash

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
sudo apt install python3-pip

# then go to the folder you got from me (i.e. folder containing this file)
pip install -r requirement.txt