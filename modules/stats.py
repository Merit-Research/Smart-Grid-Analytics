# Filename:     stats.py
# Authors:      apadin, dvorva, mjmor, mgkallit
# Start Date:   1/17/2017
# Last Update:  1/17/2017

"""Helper functions for performing statistical analysis

f1_scores - Takes as input the set of detected and known anomalies
            and returns the precision, recall, and f1-score

"""

#==================== LIBRARIES ====================#
import numpy as np


#==================== FUNCTIONS ====================#

def f1_scores(detected, ground_truth):
    """
    Calculates the f1-scores for the given sets
     - 'detected' is a set containing the timestamps for all detected anomalies
     - 'ground_truth' is a set containing the timestamps for all known anomalies
    """
    # Calculate the true positives, false positives and false negatives
    # For more information, see: https://en.wikipedia.org/wiki/Precision_and_recall
    TP = (detected & ground_truth)
    FP = float(len(detected - TP))
    FN = float(len(ground_truth - TP))
    TP = float(len(TP))
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
    return precision, recall, f1_score


def ewma(current, previous, weight):
    """Exponentially weighted moving average: z = w*z + (1-w)*z_1"""
    return weight * current + ((1.0 - weight) * previous)


def moving_average(interval, window_size):
    """
    Returns the "smoothed" version of data
    The moving average of the given interval.
    """
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    

def error_scores(targets, predictions, smoothing_win=None):
    """
    Calculates the various error values
    - targets: target values
    - predictions: predicted values
    - smoothing_win: smoothing window (number of samples)
    """
    T = len(targets)
    targets = np.array(targets).flatten()
    predictions = np.array(predictions).flatten()
    assert(len(targets) == len(predictions))

    if (smoothing_win == None):
        smoothing_win = len(targets)
    try:
        targets_smoothed = moving_average(targets, smoothing_win)
        predictions_smoothed = moving_average(predictions, smoothing_win)
    except ValueError as e:
        raise RuntimeError("Smoothing window ({}) cannot be larger than number of data points ({})".format(smoothing_win, T))

    # Prediction Mean Squared Error (smooth values)
    PMSE_smoothed = np.linalg.norm(targets_smoothed - predictions_smoothed)**2 / T
    # Prediction Mean Squared Error (raw values)
    PMSE = np.linalg.norm(targets - predictions)**2 / T
    # Relative Squared Error
    Re_MSE = np.linalg.norm(targets - predictions)**2 / np.linalg.norm(targets)**2
    # Standardise Mean Squared Error
    SMSE =  np.linalg.norm(targets - predictions)**2 / T / np.var(targets)

    return PMSE_smoothed, PMSE, Re_MSE, SMSE
    
    