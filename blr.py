# All functions and classes for running the BLR algorithm
# Filename:     algo.py
# Author(s):    apadin, with code from dvorva, mjmor, yabskbd
# Start Date:   5/9/2016

import time
import datetime as dt
import numpy as np
import pickle

from param import DATE_FORMAT
from algoFunctions import train, severityMetric, runnable


#==================== PARAMETERS ====================#
X_BACKUP_FILENAME = 'X_backup.bak'
RESULTS_FILENAME = 'results.csv'


#==================== ALGO CLASS ====================#
# This class defines the BLR algorithm and associated data manipulation.
# It is meant to act in conjunction with other programs which perform data
# collection and pass their data to Algo for analysis.
class Algo(object):

    # Constructor
    def __init__(self, granularity, training_window, forecasting_interval, num_features):

        # granularity           -> time between measurements
        #y_B matrix_length         -> number of data points to train on
        # forecasting_interval  -> number of data points between re-training sessions
        # num_features          -> number of features to train on
        self.granularity = int(granularity)
        self.granularity_in_seconds = int(granularity * 60)
        self.matrix_length = int(training_window * (60 / granularity))
        self.forecasting_interval = int(forecasting_interval * (60 / granularity))
        self.num_features = num_features

        # X matrix - each row has the feature data with the corresponding
        # power on the end
        self.X = np.zeros([self.matrix_length, self.num_features+1])

        # Regression and severity variables
        self.w_opt = []
        self.a_opt = 0
        self.b_opt = 0
        self.S_N = 0
        
        self.mu = 0 #TODO
        self.sigma = 1000
        
        # Severity parameters. Other pairs can also be used, see paper
        #self.w, self.L = (0.53, 3.714) # Most sensitive
        #self.w, self.L = (0.84, 3.719) # Medium sensitive
        self.w, self.L = (1.00, 3.719) # Least sensitive
        self.THRESHOLD = self.L * np.sqrt(self.w/(2-self.w))
        self.Sn_1 = 0
        self.alert_counter = 0
        self.init_training = False
        self.using_backup = False
        self.row_count = 0
        
        # EMA parameter
        self.alpha = 1.0

    # Read the previous training window from a backup file
    def fromBackup(self, filename=X_BACKUP_FILENAME):

        self.X_backup_file = filename
        self.using_backup = True

        # Attempt to read the given backup file name
        try:
            with open(filename, 'rb') as infile:
                X_backup = pickle.load(infile)
        except IOError:
            print "***WARNING: No training backup found.***"
        else:
            if (np.shape(X_backup) == np.shape(self.x)):
                print "Training backup file found..."
                self.X = X_backup
                self.init_training = True
            else:
                print "Unable to use training backup. Continuing analysis without backup..."
                pass

    # Add new data, train
    def run(self, new_data):
    
        if self.row_count == 0:
            self.last_avg = new_data
        else:
            new_data = (1 - self.alpha)*self.last_avg + self.alpha*new_data
            self.last_avg = new_data[:]
        
        self.addData(new_data)

        # Check if it's time to train
        if ( ((self.row_count % self.forecasting_interval) == 0) and
             ((self.row_count >= self.matrix_length) or self.init_training) ):
            self.train()

        # Check if we can make a prediction
        if self.init_training:
            prediction = self.prediction(new_data[:-1])
            target = new_data[-1]
            x_test = new_data[:-1]
            
            # Update variance (sigma)
            self.sigma = np.sqrt(1/self.b_opt + np.dot(np.transpose(x_test), 
                                                       np.dot(self.S_N, x_test)))
            
            # Catching pathogenic cases where variance gets too small
            if self.sigma < 1: 
                self.sigma = 1
                
            return target, prediction
        else:
            return new_data[-1], None
            
    # Update severity metric and check for anomaly
    # Return true if anomaly is detected, false otherwise
    def checkSeverity(self, target, prediction):
        error = prediction - target
        Sn, Zn = severityMetric(error, self.mu, self.sigma, self.w, self.Sn_1)

        # Uses two-in-a-row counter similar to branch prediction
        if np.abs(Sn) <= self.THRESHOLD:
            self.alert_counter = 0
            anomaly_found = False
        elif np.abs(Sn) > self.THRESHOLD and self.alert_counter == 0:
            #print "Severity: %.3f" %(np.abs(Sn))
            self.alert_counter = 1
            anomaly_found = False
            Sn = self.Sn_1
        elif np.abs(Sn) > self.THRESHOLD and self.alert_counter == 1:
            #print "Severity: %.3f" %(np.abs(Sn))
            Sn = 0
            anomaly_found = True
            #print "ERROR: ANOMALY"

        self.Sn_1 = Sn
        return anomaly_found

    # Add new row of data to the matrix
    def addData(self, new_data):
        assert (len(new_data) == self.num_features + 1)
        current_row = self.row_count % self.matrix_length
        self.X[current_row] = new_data
        self.row_count += 1

    # Train the model
    def train(self):
    
        # Unwrap the matrices (put the most recent data on the bottom)
        pivot = self.row_count % self.matrix_length
        data = self.X[pivot:, :self.num_features]
        data = np.concatenate((data, self.X[:pivot, :self.num_features]), axis=0)
        y = self.X[pivot:, self.num_features]
        y = np.concatenate((y, self.X[:pivot, self.num_features]), axis=0)

        if (self.init_training or runnable(data) > 0.5):
            #self.w_opt, self.a_opt, self.b_opt, self.S_N = normalTrain(data, y)
            self.w_opt, self.a_opt, self.b_opt, self.S_N = train(data, y)
            self.init_training = True
            
        # Log current training windows as pickle files
        if self.using_backup:
            with open(self.X_backup_file, 'wb') as outfile:
                pickle.dump(self.X, outfile)

    # Make a prediction based on new data
    def prediction(self, new_data):
        assert len(new_data) == len(self.w_opt)
        return max(0, np.inner(new_data, self.w_opt))
        
    # Change the severity parameters (omega w and lambda L)
    def setSeverityParameters(self, w, L):
        self.w = w
        self.L = L
        self.THRESHOLD = self.L * np.sqrt(self.w/(2-self.w))
        print "w = %.3f, L = %.3f, THRESHOLD = %.3f" % (self.w, self.L,self.THRESHOLD)
        
    def setEMAParameter(self, alpha):
        self.alpha = alpha
        print "alpha: %.3f" % alpha


