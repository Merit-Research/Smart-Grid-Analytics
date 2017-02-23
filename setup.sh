# Filename:         setup.sh
# Contributors:     apadin
# Start Date:       2017-01-12


echo "Installing Z-Way software"
wget -q -O - razberry.z-wave.me/install/v2.3.0  | bash

echo "Install python dependencies"
apt-get update
apt-get install python-pip
pip install numpy

