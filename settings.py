# Filename:     settings.py
# Authors:      apadin
# Start Date:   1/10/2017
# Last Update:  1/17/2017

"""
Helper functions for loading and saving settings. 

This library is used in conjunction with the various
scripts and programs in this repository to create a 
standardized method for running the analysis tools with 
different parameters.

- Adrian Padin, 1/10/2017
"""


#==================== LIBRARIES ====================#
import json
from collections import OrderedDict


#==================== FUNCTIONS ====================#

def load(infile):
    """Return a dictionary of settings from the given file."""
    with open(infile, 'rb') as file:
        settings = OrderedDict(json.load(file))
    return settings
        
def save(settings, outfile):
    """Save the settings dictionary to the given file."""
    with open(outfile, 'wb') as file:
        json.dump(settings, file)
