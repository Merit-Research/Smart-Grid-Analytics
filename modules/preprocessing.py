# Filename:     preprocessing.py
# Authors:      apadin, mgkallit
# Start Date:   3/7/2017
# Last Update:  3/7/2017

"""Helper functions for preprocessing data

scale_features - Scale X matrix to put values in range of about -0.5 to 0.5
auto_regression - Adds n auto-regressive features to the X matrix

"""

#==================== LIBRARIES ====================#
import numpy as np
import pandas as pd

#==================== FUNCTIONS ====================#

def scale_features(X):
    """Scale X matrix to put values in range of about -0.5 to 0.5"""
    X_scaled = (X - X.mean(0)) / (X.max(0) - X.min(0))
    return np.nan_to_num(X_scaled)
    

def add_auto_regression(X, y, n):
    """Adds n auto-regressive features to the X matrix"""
    for roll_value in xrange(n):
        y = np.roll(y, 1)
        y[0] = 0
        X = np.concatenate((X, y), 1)
    return X
    
    
def filter_low_variance(df):
    """
    Filter features with little or no variance
    Returns a new dataframe and list of features removed
    """
    removed_list = []
    for column in df.columns:
        values = df[column].values
        if (values.max() == values.min()):
            df = df.drop(column, 1)
            removed_list.append(column)
    return df, removed_list
