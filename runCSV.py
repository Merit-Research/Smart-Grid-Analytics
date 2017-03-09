#!/usr/bin/env python
# Driver for testing various prediction systems
# Filename:     driverCSV.py
# Author(s):    apadin
# Start Date:   8/12/2016
# Last Updated: 2/16/2017

#==================== LIBRARIES ====================#

import sys
import time
import csv
import argparse
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from modules.common import *
from modules.algo import Algo
from modules.preprocessing import scale_features, add_auto_regression, filter_low_variance
from modules.stats import f1_scores, error_scores
import modules.settings as settings

#==================== FUNCTIONS ====================#

def add_attacks(y, num_attacks, duration, intensity):
    """Inject attacks into the y, and return new y and affected indices"""
    if len(y) < num_attacks:
        raise RuntimeError("Too few data points for number of attacks.")

    # Pick 'num_attacks' evenly-spaced indices to attack
    delta = int(len(y) / (num_attacks + 1))
    ground_truth = set()
    for i in xrange(num_attacks):
        start = (1 + i) * delta
        end = start + duration
        y[start:end] += intensity
        ground_truth.update(range(start, end))
        
    return y, ground_truth
    

#==================== MAIN ====================#
def main(argv):

    #np.seterr(all='print')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help="name of input CSV file")
    parser.add_argument('outfile', type=str, help="name of output CSV file")
    parser.add_argument('-f', '--settings_file', type=str, help="load analysis settings from file")
    args = parser.parse_args(argv[1:])

    infile = args.infile
    outfile = args.outfile
    print ("Starting analysis on %s..." % infile)
    print ("Results will be recorded in %s..." % outfile)
    
    # Use default settings or read settings from settings file
    if (args.settings_file == None):
        settings_dict = {
            "granularity": 60,
            "training_window": 1440,
            "training_interval": 60,
            "ema_alpha": 1.0,
            "severity_omega": 1.0,
            "severity_lambda": 3.719,
            "auto_regression": 0.0
        }
    else:
        try:
            settings_dict = settings.load(args.settings_file)
        except Exception as error:
            print "Error reading settings file.", error
            print " "
            exit(1)

    training_window = int(settings_dict['training_window'])
    training_interval = int(settings_dict['training_interval'])
    ema_alpha = float(settings_dict['ema_alpha'])
    severity_omega = float(settings_dict['severity_omega'])
    severity_lambda = float(settings_dict['severity_lambda'])
    auto_regression = int(settings_dict['auto_regression'])
        
    # Collect data from CSV file
    df = pd.read_csv(infile)
    df = df.set_index(df.columns[0])
    timestamps = df.index.values
    targets = df[df.columns[-1]]
    features = df[df.columns[:-1]]
    
    # Pre-processing
    features, removed = filter_low_variance(features)
    print "\nRemoving features due to low variance:"
    for f in removed:
        print f
        
    num_features = len(features.columns) + auto_regression
    print "\nThe following {} features will be used:".format(num_features)
    for f in features.columns:
        print f
    if auto_regression > 0:
        print "Auto-regression: {}\n".format(auto_regression)
        
    # Add attacks to the data (power is last row)
    # Used for F1 calculation
    y = np.matrix(targets.values).T
    start = len(y) / 2
    y[start:], ground_truth = add_attacks(y[start:], 3, 30, 500)
    ground_truth = set([(i + start) for i in ground_truth])
    detected = set()

    X = np.matrix(features.values)
    X = scale_features(X)
    X = add_auto_regression(X, y, auto_regression)
    X = np.concatenate((X, y), 1)
    
    print ""
    
    # Initialize Algo class
    algo = Algo(num_features, training_window, training_interval)
    algo.set_severity(severity_omega, severity_lambda)
    algo.set_EWMA(ema_alpha)
            
    # Output lists
    results = [['timestamp', 'target', 'prediction', 'anomaly', 'zscore']]

    print "Num features: ", num_features
    print "w = %.3f, L = %.3f" % (severity_omega, severity_lambda)
    print "alpha: %.3f" % ema_alpha
    

    #==================== ANALYSIS ====================#
    print "Beginning analysis..."
    count = 0
    start_time = time.time()
    for count in range(len(timestamps)):

        # Get the next row of data
        cur_time = timestamps[count]

        if (count % 120 == 0):
            print "Trying time %s" % \
                dt.datetime.fromtimestamp(cur_time).strftime(DATE_FORMAT)

        new_data = np.ravel(X[count, :])
        new_data = np.around(new_data, decimals=2)
        target, prediction, anomaly, zscore = algo.run(new_data) # Magic!
        
        if (prediction != None):
            results.append([cur_time, target, prediction, anomaly, zscore])
            if anomaly:
                detected.add(count)

        count += 1

    #==================== RESULTS ====================#
    print "Runtime: %.4f" % (time.time() - start_time)
        
    # Save data for later graphing
    with open(outfile, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    # Remove headers for analysis and graphing
    timestamps, targets, predictions, anomalies, zscores = zip(*results)
    timestamps = timestamps[1:]
    targets = targets[1:]
    predictions = predictions[1:]
    anomalies = anomalies[1:]
    zscores = zscores[1:]
    
    precision, recall, f1_score = f1_scores(detected, ground_truth)
    PMSE_smoothed, PMSE, Re_MSE, SMSE = error_scores(targets, predictions) #Remove headers
    
    print "{}: {:.4f}".format("RMSE-score (smoothed)", PMSE_smoothed)
    print "{}: {:.4f}".format("RMSE-score (raw)", PMSE)
    print "{}: {:.4f}".format("Relative MSE", Re_MSE)
    print "{}: {:.4f}".format("SMSE", SMSE)
    print "{}: {:.4f}".format("Precision", precision)
    print "{}: {:.4f}".format("Recall", recall)
    print "{}: {:.4f}".format("F1-Score", f1_score)
    
    print "Ending analysis. See %s for results." % outfile
    
    plt.figure()
    
    plt.subplot(211)
    plt.plot(timestamps, targets, timestamps, predictions)
    plt.subplot(212)
    plt.plot(timestamps, zscores)
    plt.title("Targets and Predictions")
    
    """
    plt.subplot(312)
    error = [targets[i] - predictions[i] for i in range(len(targets))]
    plt.hist(np.ravel(error), 250, facecolor='green', alpha=0.75)
    plt.axis([-1000, 1000, 0, 10000])
    plt.title("Distribution of Errors")
    
    plt.subplot(313)
    plt.hist(np.ravel(pvalues), 50, facecolor='green', alpha=0.75)
    plt.title("Distribution of P-Values")
    """
    
    plt.tight_layout()
    plt.show()
    
    return results
    
    
#==================== DRIVER ====================#
if __name__ == "__main__":
    main(sys.argv)
