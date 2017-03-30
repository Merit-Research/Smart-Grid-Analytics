#!/usr/bin/env bash
# Filename:         setup.sh
# Contributors:     apadin
# Start Date:       2017-01-12

# Install necessary programs for using the Merit ISGADA code

echo "Installing Z-Way software"
#wget -q -O - razberry.z-wave.me/install/v2.3.0  | bash

echo "Installing python dependencies"
apt-get update
apt-get install libblas-dev
apt-get install liplapack-dev
apt-get install python-setuptools
apt-get install python-pip
pip install numpy
pip install scipy
pip install sklearn
pip install matplotlib
apt-get install python-qt4

while :
do
    read -r -p "Reboot now or later? [y/N] " response
    response=${response,,}   #lowercase
    if [[ "$response" = "y" ]]; then
        echo "Rebooting now..."
        reboot
    elif [[ "$response" = "n" || "$response" == "" ]]; then
        echo "Programs installed properly. Exiting now..."
        break
    fi
done

