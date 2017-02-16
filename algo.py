# Filename:     algo.py
# Authors:      apadin
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
from collections import deque

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
        if (self.train_count == self.training_interval) and \
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
