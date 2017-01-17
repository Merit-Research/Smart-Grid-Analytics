<<<<<<< HEAD
# Filename:     algo.py
# Author(s):    apadin
# Start Date:   8/12/2016
# Last Update:  1/17/2017

"""Wrapper class for processing data

The Algo class manages the collection, preprocessing,
manipulation of the data. It invokes re-training of the
model when necessary and runs the anomaly detection scheme.
"""


#==================== LIBRARIES ====================#
import sys
import numpy as np
import matplotlib.pyplot as plt

import blr
from stats import ewma


#==================== CLASSES ====================#
class Algo(object):

    def __init__(self, num_features, training_window, training_interval):
        """
        - 'num_features' is the length of the feature vector
        - 'training_window' is the number of previous data points to train on
        - 'training_interval' is the number of data points between training periods
        """

        self.num_features = num_features
        self.training_interval = training_interval
        self.training_window = training_window

        # Init sample matrix, a deque of feature vectors
        self.samples = deque(maxlen=training_window)
        self.targets = deque(maxlen=training_window)

        self.severity = blr.Severity()
        self.alpha = 1.0
        self.parameters = 0     # Training parameters
        self.train_count = 0
        self.have_trained = False
        self.pred_range = [0.0, np.inf]   # upper and lower bounds for predictions


    def run(self, sample):
        """Add a single sample to the data pool.
        The sample should be a feature vector: {x_1, x_2, x_3, ..., x_n, y}
        Where x_1->x_n are features and y is the target value
        """

        # Error checking on 'sample'
        assert(len(sample) == (self.num_features + 1))
        sample = np.array(sample).flatten()

        self.targets.append(sample[-1]) # Preserve the target value
        sample[-1] = 1                  # Constant feature, aka bias

        # Add the sample to the sample queue
        if len(self.samples) > 0:
            sample = ewma(sample, self.data_queue[-1], self.alpha)
        self.samples.append(sample)
        self.train_count += 1

        prediction = None
        anomaly = None

        # Make a prediction (if we have already trained once)
        if self.have_trained:
            prediction = sample * self.parameters   # Linear Regression model
            prediction = max(prediction, self.pred_range[0])
            prediction = min(prediction, self.pred_range[1])
            anomaly = self.severity.check(target - prediction, sample)
            #if anomaly:
            #    self.data_queue.pop()

        # Train after every 'training_interval' number of samples
        if (self.train_count == self.training_interval) and
           (len(self.samples) == training_window) :
            self.train()
            self.have_trained = True
            self.train_count = 0

        return target, prediction, anomaly


    def train(self):
        """Train the prediction and anomaly detection models"""
        X = np.matrix(self.samples)
        y = np.matrix(self.targets)
        self.parameters, alpha, beta, covariance = blr.train(X, y)
        severity.update_params(beta, covariance)


    def set_severity(self, w, L):
        """Change the severity parameters"""
        self.severity.set_wL(w, L)


    def set_EWMA(self, alpha):
        """Change the EWMA (exponential weighted moving average) weight"""
        self.alpha = alpha
=======
# Wrapper class for preprocessing data
# Filename:     algo.py
# Author(s):    apadin
# Start Date:   8/12/2016


#==================== LIBRARIES ====================#

import sys
import numpy as np

import matplotlib.pyplot as plt

from param import *
from linreg import *
from severity import Severity


#==================== CLASSES ====================#

class Algo(object):
    
    def __init__(self, num_features, training_window, training_interval):
    
        # Init data matrices
        self.num_features = num_features
        self.training_interval = training_interval
        self.training_window = training_window
        self.X = np.matrix(np.zeros((training_window, num_features)))
        self.y = np.matrix(np.zeros((training_window, 1)))
        self.theta = np.matrix(np.zeros((1, num_features)))
        
        # Counter variables
        self.row_count = 0
        self.train_count = 0
        self.init_training = False
        
        # EMA Parameter
        self.alpha = 1.0

        # Choose a regression model
        #self.regression = NeuralNetwork((num_features, int(num_features)/2, 1))
        #self.regression = LinearRegression(0.3)
        #self.regression = Poly2Regression(0.3)
        #self.regression = SklearnLinReg(0.0)
        self.regression = SklearnBLR()
        
        # Choose an anomaly detection tool
        self.severity = Severity()

    def run(self, sample):
        """Add a sample to the data pool."""
        if self.row_count == 0:
            self.last_avg = sample
        else:
            sample = (1 - self.alpha)*self.last_avg + self.alpha*sample
            self.last_avg = sample[:]
        
        assert(len(sample) == (self.num_features + 1))
        self.row_count = self.row_count % self.training_window
        self.X[self.row_count, :] = sample[:-1]
        self.y[self.row_count, 0] = sample[-1]
        self.row_count += 1
        #print self.row_count
        
        # First time training after 'training_window' samples
        if not self.init_training and self.row_count == self.training_window:
            #print "started training"
            self.init_training = True
            self.train()
            self.train_count += 1
            
        # Subsequent prediction/training after 'training_interval' samples
        elif self.init_training:
            self.train_count = self.train_count % self.training_interval
            target = sample[-1]
            prediction = self.regression.predict(sample[:-1])
            prediction = max(prediction, 0)
            
            if prediction > 12000:
                prediction = 12000
                print "whoa!!!!!"

            anomaly, pvalue = self.severity.check(target, prediction)
            
            if self.train_count == 0:
                self.train()
                
            self.train_count += 1
            return target, prediction, anomaly, pvalue
            
        return sample[-1], None, None, None
        
    def train(self):
        """Train the prediction and anomaly detection models."""
        self.regression.train(self.X, self.y)
        predictions = np.asarray([self.regression.predict(row) for row in self.X])
        targets = np.asarray(self.X[:, -1].transpose())
        self.severity.update(targets, predictions)

    def setSeverityParameters(self, *args):
        """Change the severity parameters."""
        self.severity.setSeverityParameters(*args)
        
    def setEMAParameter(self, alpha):
        """Change the EMA (exponential moving average) parameter."""
        self.alpha = alpha
        
>>>>>>> 180b14b760d397778d37ec456a28907b382f822a
