# Wrapper class for preprocessing data
# Filename:     algo.py
# Author(s):    apadin
# Start Date:   8/12/2016


#==================== LIBRARIES ====================#

import sys
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from param import *
#from linreg import *
from pybrainNN import NeuralNetwork
from severity import Severity


#==================== CLASSES ====================#

class Algo(object):
    
    def __init__(self, num_features, training_window, training_interval):
    
        # Init data matrices
        self.num_features = num_features
        self.training_interval = training_interval
        self.training_window = training_window
        self.data_queue = deque(maxlen=training_window)
        
        # Training variables
        self.train_count = 0
        self.init_training = False
        
        self.alpha = 1.0 # EMA Parameter
        self.regression = SklearnBLR() # Regression model
        self.severity = Severity(maxlen=training_window) # Anomaly detection

    def run(self, sample):
        """Add a single sample to the data pool."""
        
        assert(len(sample) == (self.num_features + 1))
        sample = np.array(sample)
        
        if len(self.data_queue) > 0:
            last_sample = self.data_queue[-1]
            sample = (1 - self.alpha)*last_sample + self.alpha*sample

        self.data_queue.append(sample)
        self.train_count += 1

        target = sample[-1]
        prediction = None
        anomaly = None
        pvalue = None
        
        # Make a prediction (if we have already trained once)
        if self.init_training:
            prediction = self.regression.predict(sample[:-1])
            prediction = max(prediction, 0)
            anomaly, pvalue = self.severity.check(target, prediction)
            #if anomaly:
            #    self.data_queue.pop()
            
        # Check if it is time to retrain
        # Train after every 'training_window' number of samples
        if self.train_count == self.training_interval:
            #print "training"
            self.train()
            self.init_training = True
            self.train_count = 0
            
        return target, prediction, anomaly, pvalue
        
    def train(self):
        """Train the prediction and anomaly detection models."""
        data_matrix = np.matrix(self.data_queue)
        X = data_matrix[:, :-1] # feature data
        y = data_matrix[:, -1]  # power data
        self.regression.train(X, y)
        
        if self.init_training:
            self.severity.update()

    def setSeverityParameters(self, *args):
        """Change the severity parameters."""
        self.severity.setSeverityParameters(*args)
        
    def setEMAParameter(self, alpha):
        """Change the EMA (exponential moving average) parameter."""
        self.alpha = alpha
        