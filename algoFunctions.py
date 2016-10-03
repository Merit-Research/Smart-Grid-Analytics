# Helper functions for algorithm
# Filename:     algoRunCSV.py
# Author(s):    dvorva, mjmor
# Start Date:   4/30/2016

import os
import sys
import time
import numpy as np
import scipy as sp
import scipy.stats
import csv
import json
from urllib import urlopen

debug = 1


# Returns the moving average of the given interval
def movingAverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    

# Returns the percentage of valid data points in the array
# Used to determine of the data is valid enough to train on
def runnable(arrayIn):
    countAll = 0
    countValid = 0
    for row in arrayIn:
        for datum in row:
            countAll += 1
            if int(datum) is not -1:
                countValid += 1
    return float(countValid)/countAll
    
    
# Finds the optimal weight matrix using the Normal Function
# theta = pinv(X' * X) * X' * y
def normalTrain(X, y):
    X = np.matrix(X)
    y = np.matrix(y).transpose()
    X_t = X.transpose()
    w_opt = np.linalg.pinv(X_t * X) * X_t * y
    w_opt = [float(i) for i in w_opt]
    #print w_opt
    return w_opt, 1.0, 1.0, 1.0
    
    
# This function is used for training our Bayesian model
# Returns the regression parameters w_opt, and alpha, beta, S_N
# needed for the predictive distribution    
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

    
# Returns the Variance (Sn) and Z-Scores (Zt) of the EWMA control char
# as described by the paper
def severityMetric(error, mu, sigma, w, Sn_1):

    # Left-tailed
    if error < mu:
        p_value = sp.stats.norm.cdf(error, mu, sigma)
        Zt = sp.stats.norm.ppf(p_value) # inverse of cdf N(0,1)
        
    # Right-tailed
    else:
        p_value = 1 - sp.stats.norm.cdf(error, mu, sigma)
        Zt = sp.stats.norm.ppf(1-p_value) # inverse of cdf N(0,1)

    if Zt > 10:
        Zt = 10
    elif Zt < -10:
        Zt = -10

    Sn = (1-w)*Sn_1 + w*Zt

    if debug:
        if np.abs(Zt) > 90:
            print "Error = %d, p-value=%.3f, Z-score=%.3f, Sn_1=%.2f, Sn=%.2f " % (error, p_value, Zt, Sn_1, Sn)
        elif np.abs(Zt) < 0.005:
            print "Error = %d, p-value=%.3f, Z-score=%.3f, Sn_1=%.2f, Sn=%.2f " % (error, p_value, Zt, Sn_1, Sn)

    return Sn, Zt

            
# Calculates the f1-scores for the given sets
# 'detected' is a set containing the timestamps for all detected anomalies
# 'ground_truth' is a set containing the timestamps for all known anomalies
def f1_scores(detected, ground_truth):

    # Calculate the True Positives, False Positives and False Negatives
    # For more information, see: https://en.wikipedia.org/wiki/Precision_and_recall
    TP = (detected & ground_truth)
    FP = float(len(detected - TP))
    FN = float(len(ground_truth-TP))
    TP = float(len(TP))
    print "TP: {}, FP: {}, FN: {}".format(TP,FP,FN)

    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = float('nan')

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = float('nan')

    try:
        f1_score = (2*TP)/((2*TP) + FP + FN)
    except ZeroDivisionError:
        f1_score = float('nan')

    print "Precision: {}\nRecall: {}\nF1 Score: {}".format(precision, recall, f1_score)
    return precision, recall, f1_score
        

# Calculates the various error values
# 'y_target' is a list of target power values in Watts
# 'y_predict' is a list of predicted power values in Watts
# 'smoothing_win' is the smoothing window in minutes
def print_stats(y_target, y_predict, smoothing_win=120):

    T = len(y_target)
    y_target = np.array(y_target, np.float)
    y_predict = np.array(y_predict, np.float)

    try:
        y_target_smoothed = movingAverage(y_target, smoothing_win)
        y_predict_smoothed = movingAverage(y_predict, smoothing_win)
    except ValueError as e:
        print repr(e)
        print "Error: Smoothing window cannot be larger than number of data points"
        y_target_smoothed = movingAverage(y_target, 1)
        y_predict_smoothed = movingAverage(y_predict, 1)

    # Prediction Mean Squared Error (smooth values)
    PMSE_score_smoothed = np.linalg.norm(y_target_smoothed-y_predict_smoothed)**2 / T
    # Prediction Mean Squared Error (raw values)
    PMSE_score = np.linalg.norm(y_target - y_predict)**2 / T
    # Relative Squared Error
    Re_MSE = np.linalg.norm(y_target-y_predict)**2 / np.linalg.norm(y_target)**2
    # Standardise Mean Squared Error
    SMSE =  np.linalg.norm(y_target-y_predict)**2 / T / np.var(y_target)

    print "---------------------------------------------------------------------------"
    print "%20s |%20s |%15s |%10s "  % ("RMSE-score (smoothed)", "RMSE-score (raw)", "Relative MSE", "SMSE")
    print "%20.3f  |%20.3f |%15.3f |%10.3f " % (np.sqrt(PMSE_score_smoothed), np.sqrt(PMSE_score), Re_MSE, SMSE)
    print "---------------------------------------------------------------------------"
    
    
def readResults(csvfile):
    """Retrieve data in file given by 'csvfile'."""
    
    with open(csvfile, 'rb') as infile:
        reader = csv.reader(infile)
        reader.next()
        return zip(*reader)


def writeResults(csvfile, results):
    """Save 'results' data in file given by 'csvfile'."""
    
    with open(csvfile, 'wb') as outfile:
        writer = csv.writer(outfile)
        row_list = zip(*results) #Get columns out of rows
        for row in row_list:
            writer.writerow(row)
