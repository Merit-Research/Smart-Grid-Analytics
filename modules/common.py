# Filename:     param.py
# Author(s):    apadin
# Start Date:   6/24/2016
# Last Update:  1/17/2017

"""Data and setup common to all modules

This file consolidates all data and setup commands which must be shared across
the various programs in this repository.
    
"""



DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ICON_FILE = 'app/merit_icon.png'
LOG_FILE = '/var/log/sequential.log'

# Logging information
import logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
