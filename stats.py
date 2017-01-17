# Filename:     stats.py
# Authors:      apadin, dvorva, mjmor, mgkallit
# Start Date:   1/17/2017
# Last Update:  1/17/2017


"""
stats.py - Helper functions for performing statistical analysis

Contains functions for calculating F1 scores and error scores

f1_scores - Takes as input the set of detected and known anomalies
            and returns the precision, recall, and f1-score

"""

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
    
    
        
        