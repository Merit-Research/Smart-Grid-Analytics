<<<<<<< HEAD
# Filename:     blr.py
# Author(s):    apadin, dvorva, mjmor, mgkallit
# Start Date:   1/17/2017


"""
blr.py - Functions for training the Bayesian model

train - Accepts as input a training set X and labels y
        and returns the optimal parameters and hyper-parameters

"""


#==================== LIBRARIES ====================#

import numpy as np
import scipy as sp
import scipy.stats

from stats import ewma


#==================== FUNCTIONS ====================#

# This function is used for training our Bayesian model
# Returns the regression parameters w_opt, and alpha, beta, S_N
# needed for the predictive distribution
# See p.152 of Bishop's "Pattern Recognition and Machine Learning"
def train(X, y):

    Phi = X # the measurement matrix of the input variables x (i.e., features)
    t   = y # the vector of observations for the target variable
    (N, M) = np.shape(Phi)
    # Init values for  hyper-parameters alpha, beta
    alpha = 5*10**(-3)
    beta = 5
    max_iter = 100
    k = 0

    PhiT_Phi = np.dot(np.transpose(Phi), Phi)
    s = np.linalg.svd(PhiT_Phi, compute_uv=0) # Just get the vector of singular values s

    ab_old = np.array([alpha, beta])
    ab_new = np.zeros((1,2))
    tolerance = 10**-3
    while( k < max_iter and np.linalg.norm(ab_old-ab_new) > tolerance):
        k += 1
        try:

            S_N = np.linalg.pinv(alpha*np.eye(M) + beta*PhiT_Phi)
        except np.linalg.LinAlgError as err:
            print  "******************************************************************************************************"
            print "                           ALERT: LinearAlgebra Error detected!"
            print "      CHECK if your measurement matrix is not leading to a singular alpha*np.eye(M) + beta*PhiT_Phi"
            print "                           GOODBYE and see you later. Exiting ..."
            print  "******************************************************************************************************"
            sys.exit(-1)

        m_N = beta * np.dot(S_N, np.dot(np.transpose(Phi), t))
        gamma = sum(beta*s[i]**2 /(alpha + beta*s[i]**2) for i in range(M))
        #
        # update alpha, beta
        #
        ab_old = np.array([alpha, beta])
        alpha = gamma /np.inner(m_N,m_N)
        one_over_beta = 1/(N-gamma) * sum( (t[n] - np.inner(m_N, Phi[n]))**2 for n in range(N))
        beta = 1/one_over_beta
        ab_new = np.array([alpha, beta])

    S_N = np.linalg.pinv(alpha*np.eye(M) + beta*PhiT_Phi)
    m_N = beta * np.dot(S_N, np.dot(np.transpose(Phi), t))
    w_opt = m_N

    return (w_opt, alpha, beta, S_N)
    

#==================== CLASSES ====================#
    
class Severity(object):
    
    def __init__(self, w=0.25, L=3, alert_count=2):
        self.beta = 1.0         # noise precision constant (determined by training)
        self.covariance = 0     # covariance (determined by training)
        self.avg_zscore = 0     # last average for EWMA chart
        self.alert_count = 0    # Number of alerts in a row
        self.set_wL(w, L)
        self.ALERT_THRESH = alert_count
    
    def update_params(beta, covariance):
        self.beta = beta
        self.covariance = covariance
        
    def set_wL(w, L):
        self.w = w  # EWMA weight
        self.L = L  # Std dev limit
        self.Z_THRESH = L * np.sqrt(w/(2-w))       

    def check(self, error, x):
        mu = 0
        beta = self.beta
        S_N = self.covariance
        x = np.matrix(x).flatten()
        
        sigma = np.sqrt(1.0/beta + np.dot(x.T, np.dot(S_N, x)))
    
        # Left-tailed
        if error < mu:
            p_value = sp.stats.norm.cdf(error, mu, sigma)
            zscore = sp.stats.norm.ppf(p_value) # inverse of cdf N(0,1)
            
        # Right-tailed
        else:
            p_value = 1 - sp.stats.norm.cdf(error, mu, sigma)
            zscore = sp.stats.norm.ppf(1 - p_value) # inverse of cdf N(0,1)
            
        # Keep the zscore bounded
        zscore = min(10, abs(zscore))

        # Exponentially weighted moving average (EWMA)
        self.avg_zscore = ewma(zscore, self.avg_zscore, self.w)
        
        # Detect anomalies with alert counter
        # A single alert is raised if the Z score is over the threshold
        if (self.avg_zscore > self.Z_THRESH):
            self.alert_count = self.alert_count + 1
        else:
            self.alert_count = 0

        # If several alerts are raised in succession, an anomly is reported
        if (self.alert_count >= self.ALERT_THRESH):
            return True
        else:
            return False
=======
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


>>>>>>> 180b14b760d397778d37ec456a28907b382f822a
