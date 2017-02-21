#!/usr/bin/env python
# Filename:     driverSeq.py
# Authors:      apadin, based on work by dvorva, yabskbd, mjmor, and mgkallit
# Start Date:   2017-01-10

"""Driver program for running the sequential BLR

This program is intended to replace all versions of 
pi_seq_BLR* currently floating around. It was created
in order to help organize the repository after it became
public so that other viewers could better understand the
analysis process. It also implements several improvements
which would have been difficult to include in the 
existing program. This includes:
 * Use of pandas for data manipulation and file I/O
 * Deques over lists or numpy arrays to hold the data
 * The Algo class to handle the BLR analysis
 * Use of json library to save and load settings
 * Simplification of argparse options
 * Better abstraction of data collection
 * Separation of training backups into separate module

- Adrian Padin, 1/10/2017

"""

#==================== LIBRARIES ====================#
import os
import sys
import time
import argparse
import numpy as np

from common import *
import settings
import zway
from datalog import Datalog
from algo import Algo


#==================== FUNCTIONS ====================#

def collect_features(zserver, sound=False):
    feature_list = []
    for key in zserver.device_IDs():
        feature_list.append(zserver.get_data(key))
    feature_list.append(np.random.rand())
    return np.array(feature_list)

    
#==================== MAIN ====================#
def main(argv):
    
    #===== Initialization =====#
    prefix = 'pi_seq_test'
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('hostname', type=str)
    parser.add_argument('settings_file', type=str)
    parser.add_argument('-s', '--sound', action='store_true', help="use sound as a feature in analysis")
    parser.add_argument('-b', '--backup', action='store_true', help="start training on backup data")
    parser.add_argument('-t', '--time_allign', action='store_true', help="collect data at times which are multiples of the granularity")
    args = parser.parse_args(argv[1:])
        
    # Initialize Zway server
    host = args.hostname
    zserver = zway.Server(host)

    # Read settings from settings file
    try:
        settings_dict = settings.load(args.settings_file)
    except Exception as error:
        print "Error reading settings file.", error
        print " "
        exit(1)
        
    feature_list = zserver.device_IDs()

    # Initialize Algo class
    granularity = int(settings_dict['granularity'])
    training_window = int(settings_dict['training_window'])
    training_interval = int(settings_dict['training_interval'])
    ema_alpha = float(settings_dict['ema_alpha'])
    severity_omega = float(settings_dict['severity_omega'])
    severity_lambda = float(settings_dict['severity_lambda'])
    auto_regression = int(settings_dict['auto_regression'])
    num_features = len(feature_list)
    
    print "Num features: ", num_features
    print "w = %.3f, L = %.3f" % (severity_omega, severity_lambda)
    print "alpha: %.3f" % ema_alpha

    
    algo = Algo(num_features, training_window, training_interval)
    algo.setSeverityParameters(severity_omega, severity_lambda)
    algo.setEMAParameter(ema_alpha)
    
    # Two Datalogs: one for data and one for results
    data_log = Datalog(prefix, feature_list)
    results_header = ['power', 'prediction', 'anomaly']
    results_log = Datalog(prefix + '_results', results_header)
    
    # Timing procedure
    granularity = settings_dict['granularity'] * 60
    granularity = 10
    goal_time = int(time.time())
    if args.time_allign:
        goal_time += granularity - (int(time.time()) % granularity)
    
    #===== Analysis =====#
    while(True):
    
        # Timing procedures
        while goal_time > time.time():
            time.sleep(0.2)
        goal_time = goal_time + granularity
        
        # Data collection
        features = collect_features(zserver)
        data_log.log(features.tolist(), goal_time)
        print "Sample recorded at {}".format(goal_time)
        
        # Data analysis and recording results
        target, pred = algo.run(features)
        if (pred != None):
            anomaly = algo.checkSeverity(target, pred)
            results_log.log([target, pred, float(anomaly)])
            print target, pred, anomaly
            print "theta", algo.w_opt
        else:
            print target, predwui

        
    # Clean-up if necessary

        
#==================== DRIVER ====================#
if __name__ == "__main__":
    main(sys.argv)
    
