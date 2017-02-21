#!/usr/bin/python -O
# Driver for testing various prediction systems
# Filename:     driverCSV.py
# Author(s):    apadin
# Start Date:   8/12/2016
# Last Updated: 2/16/2017

#==================== LIBRARIES ====================#

import sys
import time
import datetime as dt
import numpy as np
import csv

import matplotlib.pyplot as plt

from common import *
from algo import Algo
from stats import f1_scores, error_scores


#==================== FUNCTIONS ====================#

def add_attacks(data, num_attacks, duration, intensity):
    """Inject attacks into the data, and return new data and affected indices"""
    if len(data) < num_attacks:
        raise RuntimeError("Too few data points for number of attacks.")

    # Pick 'num_attacks' evenly-spaced indices to attack
    delta = int(len(data) / (num_attacks + 1))
    ground_truth = set()
    for i in xrange(num_attacks):
        start = (1 + i) * delta
        end = start + duration
        data[start:end] += intensity
        ground_truth.update(range(start, end))
        
    return ground_truth
    

#==================== MAIN ====================#
def main(argv):

    #np.seterr(all='print')

    try:
        infile = argv[1]
        outfile = argv[2]
    except Exception:
        raise RuntimeError("usage: python " + argv[0] + " <infile> <outfile>")

    print ("Starting analysis on %s..." % infile)
    print ("Results will be recorded in %s..." % outfile)
    
    # Collect data from CSV file
    data_array = np.loadtxt(infile, delimiter=',', skiprows=1)
    
    # Add attacks to the data (power is last row)
    # TODO: All the copying may not be necessary
    #ground_truth = add_attacks(data_array[:, -1], 5, 60, 3000)
        
    # Output lists
    results = [['timestamp', 'target', 'prediction', 'anomaly']]

    # Parameters
    num_features = data_array.shape[1] - 2 #subtract 2 for timestamp and target
    training_window = 24
    training_interval = 1
    
    algo = Algo(num_features, training_window*60, training_interval*60)
        
    # EWMA additions
    # alpha is adjustable on a scale of (0, 1]
    # The smaller value of alpha, the more averaging takes place
    # A value of 1.0 means no averaging happens
    #alpha = float(raw_input('Enter Value of alpha:'))
    #algo.set_EWMA(alpha=1.0)
    #algo.set_EWMA(alpha=0.73)
    algo.set_EWMA(alpha=1.0)
    
    #Recomended Severity Parameters from Paper
    #algo.set_severity(w=0.53, L=3.714) # Most sensitive
    algo.set_severity(w=0.84, L=3.719) # Medium sensitive
    #algo.set_severity(w=1.00, L=3.719) # Least sensitive 
    #algo.set_severity(1, 2) # Custom senstivity
    
    # Used for F1 calculation
    detected = set()

    #==================== ANALYSIS ====================#
    print "Beginning analysis..."
    count = 0
    start_time = time.time()
    for row in data_array:

        # Get the next row of data
        cur_time = int(row[0])
        if (count % 240) == 0:
            print "Trying time %s" % \
                dt.datetime.fromtimestamp(cur_time).strftime(DATE_FORMAT)

        new_data = np.asarray(row[1:], np.float)
        new_data = np.around(new_data, decimals=2)
        target, prediction, anomaly = algo.run(new_data) # Magic!
        
        if prediction != None:
            results.append([cur_time, target, float(prediction), float(anomaly)])
            if anomaly:
                detected.add(count)

        count += 1

    #==================== RESULTS ====================#
        
    # Save data for later graphing
    with open(outfile, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    # Remove headers for analysis and graphing
    timestamps, targets, predictions, anomalies = zip(*results)
    
    #f1_scores(detected, ground_truth) # Comment out if F1 Score not desired
    PMSE_smoothed, PMSE, Re_MSE, SMSE = error_scores(targets[1:], predictions[1:]) #Remove headers
    print PMSE_smoothed, PMSE, Re_MSE, SMSE
    print "Runtime: %.4f" % (time.time() - start_time)
    print "Ending analysis. See %s for results." % outfile
    
    """
    plt.figure()
    
    plt.subplot(311)
    plt.plot(timestamps, targets, timestamps, predictions)
    plt.title("Targets and Predictions")

    plt.subplot(312)
    error = [targets[i] - predictions[i] for i in range(len(targets))]
    plt.hist(np.ravel(error), 250, facecolor='green', alpha=0.75)
    plt.axis([-1000, 1000, 0, 10000])
    plt.title("Distribution of Errors")
    
    plt.subplot(313)
    plt.hist(np.ravel(pvalues), 50, facecolor='green', alpha=0.75)
    plt.title("Distribution of P-Values")
    
    plt.tight_layout()
    plt.show()
    """
    
    return results
    
    
#==================== DRIVER ====================#
if __name__ == "__main__":
    main(sys.argv)
