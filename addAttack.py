# Functions for simulating attacks on power data
# Filename:     severity.py
# Author:       apadin
# Start Date:   8/17/2016

import numpy as np
import time

# Inject attacks into the given data
# Return new data and affected indices
def addAttacks(data, num_attacks, duration, intensity):

    data = data[:]
    if len(data) < num_attacks:
        raise RuntimeError("Too few data for number of attacks.")

    # Pick 'num_attacks' random indices to attack
    # Not actually random tho
    delta = int(len(data) / (num_attacks + 1))
    
    ground_truth = set()
    
    for i in xrange(num_attacks):
        start = (1 + i) * delta
        end = start + duration
        data[start:end] += intensity
        ground_truth.update(range(start, end))
        
    return data, ground_truth
    

    