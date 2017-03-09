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
        num_features: the length of the feature vector
        training_window: the number of previous data points to train on
        training_interval: the number of data points between training periods
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
        """
        Add a single sample to the data pool.
        The sample should be a feature vector: {x_1, x_2, x_3, ..., x_n, y}
        Where x_1->x_n are features and y is the target value
        """
        try:
            assert(len(sample) == (self.num_features + 1))  # Input check
        except AssertionError:
            raise RuntimeError("sample length {} does not match number of features {}".format(len(sample), self.num_features + 1))
        sample = np.array(sample).flatten()
        target = sample[-1]
        prediction = None
        anomaly = None
        p_value = None
        self.targets.append(target) # Preserve the target value
        sample[-1] = 1              # Constant feature, aka bias

        # Add the sample to the sample queue
        if len(self.samples) > 0:
            sample = ewma(sample, self.samples[-1], self.alpha)
        self.samples.append(sample)
        self.train_count += 1

        # Make a prediction (if we have already trained once)
        if self.have_trained:
            prediction = float(np.dot(self.parameters, sample))
            prediction = np.clip(prediction, self.pred_range[0], self.pred_range[1])
            anomaly, p_value = [float(i) for i in self.severity.check(target - prediction, sample)]
            #if anomaly:
            #    self.data_queue.pop()

        # Train after every 'training_interval' number of samples
        #print self.train_count, self.training_interval
        #print len(self.samples), self.training_window
        #raw_input('')
        if (self.train_count >= self.training_interval) and \
           (len(self.samples) >= self.training_window) :
            self.train()
            self.have_trained = True
            self.train_count = 0

        return target, prediction, anomaly, p_value

    def train(self):
        """Train the prediction and anomaly detection models"""
        X = np.matrix(self.samples)
        y = np.array(self.targets).flatten()
        w_opt, alpha, beta, S_N = blr.sklearn_train(X, y)
        #w_opt, alpha, beta, S_N = blr.train(X, y)
        self.parameters = w_opt.flatten()
        
        #covariance = np.linalg.pinv(alpha*np.eye(M) + beta*PhiT_Phi)
        self.severity.update_params(beta, S_N)

    def set_severity(self, w, L):
        """Change the severity parameters"""
        self.severity.set_wL(w, L)

    def set_EWMA(self, alpha):
        """Change the EWMA (exponential weighted moving average) weight"""
        self.alpha = alpha
