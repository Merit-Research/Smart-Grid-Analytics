# Classes for assesing severity and detecting anomalies
# Filename:     severity.py
# Author:       apadin
# Start Date:   8/17/2016


#==================== LIBRARIES ====================#

import sys
import numpy as np
from collections import deque

from scipy.optimize import minimize
from scipy.stats import norm
#from matplotlib import pyplot as plt


#==================== CLASSES ====================#

class Severity(object):

    """Anomaly detector which uses simple error and z-score
       to determine if a measurement classifies as an anomaly"""

    def __init__(self, maxlen, threshold=0.01):
        """Initialize the severity detector."""
        self.error_queue = deque(maxlen=maxlen)
        self.mean = None
        self.std = None
        self.threshold = threshold;
        self.counter = 0
        self.alpha = 0.3
        self.last_p = 0.5
        
    def update(self):
        """Determines the mean and standard deviation of the training
           set to determine what is 'normal' behavior."""
        self.mean = np.mean(self.error_queue)
        self.std = np.std(self.error_queue)
        if (self.std < 1.0): self.std = 1.0  # Make sure variance is non-zero
        return self.std, self.mean
        
    def check(self, target, prediction):
        """Returns 1 if this measurement is an anomaly, 0 otherwise.
           If the classifier does not have enough information yet, return 0."""
        error = (target - prediction)
        self.error_queue.append(error)
        
        # If there is insufficient data, stop and return 0
        if self.mean == None:
            return 0, 0.5
        
        # Calculate pvalue based on mean and std deviation
        pvalue = norm.sf(error, self.mean, self.std)
        if pvalue > 0.5: pvalue = 1 - pvalue
            
        self.last_p = (1.0 - self.alpha)*self.last_p + self.alpha*pvalue
        
        anomaly = 0
        
        # Uses two-in-a-row counter similar to branch prediction
        if self.last_p >= self.threshold:
            self.counter = 0
        else:
            #self.error_queue.pop()
            if self.counter < 1:
                self.counter += 1
            else:
                anomaly = 1

        return anomaly, self.last_p
        
    def setSeverityParameters(self, threshold):
        """Set the threshold parameter of the detector."""
        self.threshold = threshold;
        

       
    
    
    
    
    
    
    
    
    
    
    