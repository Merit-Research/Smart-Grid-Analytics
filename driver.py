<<<<<<< HEAD
#!/usr/bin/python -O
# Driver for testing various prediction systems
# Filename:     driver.py
# Author(s):    apadin
# Start Date:   8/12/2016


#==================== LIBRARIES ====================#

import sys
import time
import datetime as dt
import numpy as np
import csv

import matplotlib.pyplot as plt

#from algo import Algo
from algo2 import Algo
from param import DATE_FORMAT
from algoFunctions import f1_scores, print_stats


#==================== FUNCTIONS ====================#

# Inject attacks into the given data
# Return new data and affected indices
def addAttacks(data, num_attacks, duration, intensity):

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
    with open(infile, 'rb') as csvfile:
        data_array = np.loadtxt(csvfile, delimiter=',', skiprows=1)
        
    # Add attacks to the data (power is last row)
    # TODO: All the copying may not be necessary
    ground_truth = addAttacks(data_array[:, -1], 5, 60, 3000)
        
    # Output lists
    timestamps  = ['Timestamp']
    targets     = ['Target']
    predictions = ['Prediction']
    anomalies   = ['Anomaly']
    pvalues     = ['P-Value']

    # Parameters
    training_window = 24
    training_interval = 1
    
    algo = Algo(len(data_array[0, :])-2, training_window*60, training_interval*60)
        
    # EWMA additions
    # alpha is adjustable on a scale of (0, 1]
    # The smaller value of alpha, the more averaging takes place
    # A value of 1.0 means no averaging happens
    #alpha = float(raw_input('Enter Value of alpha:'))
    #algo.setEMAParameter(alpha=1.0)
    #algo.setEMAParameter(alpha=0.73)
    algo.setEMAParameter(alpha=1.0)
    
    #Recomended Severity Parameters from Paper
    #algo.setSeverityParameters(w=0.53, L=3.714) # Most sensitive
    #algo.setSeverityParameters(w=0.84, L=3.719) # Medium sensitive
    #algo.setSeverityParameters(w=1.00, L=3.719) # Least sensitive 
    algo.setSeverityParameters(0.01) # Custom senstivity
    
    # Used for F1 calculation
    detected = set()

    #==================== ANALYSIS ====================#
    print "Beginning analysis..."
    count = 0
    stopwatch = time.time()
    for row in data_array:

        # Read new data from file
        cur_time = int(row[0])
        if (count % 240) == 0:
            print "Trying time %s" % \
                dt.datetime.fromtimestamp(cur_time).strftime(DATE_FORMAT)

        new_data = np.asarray(row[1:], np.float)
        new_data = np.around(new_data, decimals=2)
        target, prediction, anomaly, pvalue = algo.run(new_data) # Magic!

        if prediction != None:

            if anomaly:
                detected.add(count)

            timestamps.append(cur_time)
            targets.append(target)
            predictions.append(prediction)
            anomalies.append(anomaly)
            pvalues.append(pvalue)

        count += 1

    #==================== RESULTS ====================#
        
    # Save data for later graphing
    results = timestamps, targets, predictions, anomalies, pvalues
    with open(outfile, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        rows = zip(*results) #Get rows out of columns
        writer.writerows(rows)

    # Remove headers for analysis and graphing
    timestamps = timestamps[1:]
    targets = targets[1:]
    predictions = predictions[1:]
    anomalies = anomalies[1:]
    pvalues = pvalues[1:]

    f1_scores(detected, ground_truth) # Comment out if F1 Score not desired
    print_stats(targets, predictions) #Remove header
    print "Runtime: %.4f" % (time.time() - stopwatch)
    print "Ending analysis. See %s for results." % outfile
    
    
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
    
    
    return results
    
    
#==================== DRIVER ====================#
if __name__ == "__main__":
    main(sys.argv)
=======
#!/usr/bin/python -O
# Driver for testing various prediction systems
# Filename:     driver.py
# Author(s):    apadin
# Start Date:   8/12/2016


#==================== LIBRARIES ====================#

import sys
import time
import datetime as dt
import numpy as np
import csv

import matplotlib.pyplot as plt

#from algo import Algo
from algo2 import Algo
from param import DATE_FORMAT
from algoFunctions import f1_scores, print_stats


#==================== FUNCTIONS ====================#

# Inject attacks into the given data
# Return new data and affected indices
def addAttacks(data, num_attacks, duration, intensity):

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
    with open(infile, 'rb') as csvfile:
        data_array = np.loadtxt(csvfile, delimiter=',', skiprows=1)
        
    # Add attacks to the data (power is last row)
    # TODO: All the copying may not be necessary
    ground_truth = addAttacks(data_array[:, -1], 5, 60, 3000)
        
    # Output lists
    timestamps  = ['Timestamp']
    targets     = ['Target']
    predictions = ['Prediction']
    anomalies   = ['Anomaly']
    pvalues     = ['P-Value']

    # Parameters
    training_window = 24
    training_interval = 1
    
    algo = Algo(len(data_array[0, :])-2, training_window*60, training_interval*60)
        
    # EWMA additions
    # alpha is adjustable on a scale of (0, 1]
    # The smaller value of alpha, the more averaging takes place
    # A value of 1.0 means no averaging happens
    #alpha = float(raw_input('Enter Value of alpha:'))
    #algo.setEMAParameter(alpha=1.0)
    #algo.setEMAParameter(alpha=0.73)
    algo.setEMAParameter(alpha=1.0)
    
    #Recomended Severity Parameters from Paper
    #algo.setSeverityParameters(w=0.53, L=3.714) # Most sensitive
    #algo.setSeverityParameters(w=0.84, L=3.719) # Medium sensitive
    #algo.setSeverityParameters(w=1.00, L=3.719) # Least sensitive 
    algo.setSeverityParameters(0.01) # Custom senstivity
    
    # Used for F1 calculation
    detected = set()

    #==================== ANALYSIS ====================#
    print "Beginning analysis..."
    count = 0
    stopwatch = time.time()
    for row in data_array:

        # Read new data from file
        cur_time = int(row[0])
        if (count % 240) == 0:
            print "Trying time %s" % \
                dt.datetime.fromtimestamp(cur_time).strftime(DATE_FORMAT)

        new_data = np.asarray(row[1:], np.float)
        new_data = np.around(new_data, decimals=2)
        target, prediction, anomaly, pvalue = algo.run(new_data) # Magic!

        if prediction != None:

            if anomaly:
                detected.add(count)

            timestamps.append(cur_time)
            targets.append(target)
            predictions.append(prediction)
            anomalies.append(anomaly)
            pvalues.append(pvalue)

        count += 1

    #==================== RESULTS ====================#
        
    # Save data for later graphing
    results = timestamps, targets, predictions, anomalies, pvalues
    with open(outfile, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        rows = zip(*results) #Get rows out of columns
        writer.writerows(rows)

    # Remove headers for analysis and graphing
    timestamps = timestamps[1:]
    targets = targets[1:]
    predictions = predictions[1:]
    anomalies = anomalies[1:]
    pvalues = pvalues[1:]

    f1_scores(detected, ground_truth) # Comment out if F1 Score not desired
    print_stats(targets, predictions) #Remove header
    print "Runtime: %.4f" % (time.time() - stopwatch)
    print "Ending analysis. See %s for results." % outfile
    
    
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
    
    
    return results
    
    
#==================== DRIVER ====================#
if __name__ == "__main__":
    main(sys.argv)
>>>>>>> 180b14b760d397778d37ec456a28907b382f822a
