#!/usr/bin/python -O

# Version of pi_seq_BLR compatible with app.py
# Filename:     pi_seq_BLR_AVG.py
# Author(s):    apadin
# Start Date:   6/8/2016


############################## LIBRARIES ##############################
import datetime as dt
import time
import sys
import json
import logging
import numpy as np
from grapher import DATE_FORMAT
from algoRunFunctions import train, severityMetric
from get_data import get_data, get_power
from zwave_api import ZWave
import pickle


############################## PARAMTERS ##############################
Xdata_LOG_FILENAME = "X_DATA2.bak"
Xog_LOG_FILENAME = "Xog_DATA2.bak"
LOG_FILENAME = "/var/log/sequential_predictions2.log"

THRESHOLD = 10000


############################## ANALYZER ##############################
def analyze(granularity, training_window, forecasting_interval, queue):


    ############################## INITIALIZE ##############################
    print ("Starting \"Merit Energy Analysis\" with settings: %d %d %d" %
           (granularity, training_window, forecasting_interval))

    # Logging analysis results
    FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=LOG_FILENAME,
                        level=logging.DEBUG,
                        format=FORMAT,
                        datefmt=DATE_FORMAT)

    logging.info("Starting \"Merit Energy Analysis\" with settings: %d %d %d" %
                 (granularity, training_window, forecasting_interval))

    # Initialize Zwave server to collect sensor data
    with open("./config/config.json") as config_fh:
        config_dict = json.load(config_fh)
    with open("./config/sensors.json") as device_fh:
        device_dict = json.load(device_fh)
    ZServer = ZWave(config_dict["z_way_server"]["host"],
                         config_dict["z_way_server"]["port"],
                         device_dict)

    # num_sensors           -> Number of sensors in ZWave network
    # matrix_length         -> Number of rows in data matrix (X)
    # forecasting_interval  -> Time between training sessions
    # granularity           -> Time between sensor measurements
    num_sensors = len(ZServer.get_data_keys())
    matrix_length = training_window * (60 / granularity)
    forecasting_interval = forecasting_interval * (60 / granularity)
    granularity_in_seconds = granularity * 60

    # Number of counts to average over
    avg_over = 5

    # Load backup files if possible
    X = np.zeros([matrix_length, num_sensors+1])
    X_og = np.zeros([avg_over, num_sensors+1])
    init_training = False
    try:
        logged_Xdata = pickle.load(open(Xdata_LOG_FILENAME, 'r'))
        logged_Xog = pickle.load(open(Xog_LOG_FILENAME, 'r'))
    except IOError:
        logging.warning("One or more backup files not found. Continuing without backups.")
    else:
        if (np.shape(logged_Xdata) == (matrix_length, num_sensors+1) and
            np.shape(logged_Xog) == (avg_over, num_sensors+1)):

            logging.info("Backup files found. Using backups.")
            X = logged_Xdata
            X_og = logged_Xog
            init_training = True
        else:
            logging.warning("Backup files found but not properly formatted. Continuing without backups.")

    # Prepare the graphing arrays
    y_target, y_predict, y_time = [], [], []

    row_count = 0

    # Prepare the timer
    goal_time = time.time()
    goal_time = goal_time - (goal_time % 60)

    ############################## ANALYZE ##############################
    while True:

        # Record the time of the next iteration
        goal_time += granularity_in_seconds
        
        # Wake up periodically to check time and kill_flag
        while goal_time > time.time():
            time.sleep(0.1)
            
            '''
            app.lock.acquire()
            if app.kill_flag:
                app.lock.release()

                print "exiting program"
                logging.error("Analysis ended by user. Ending program.")
                sys.exit(0)
                
            app.lock.release()
            '''

        print "getting data"

        # Retrieve sensor data from ZServer
        try:
            new_data = get_data(ZServer)
        except:
            logging.error("ZServer Connection Lost. Ending program.")
            exit(1)

        print new_data

        # Retrieve energy usage reading
        new_power = float(get_power(config_dict))

        # Update X and X_og
        X_row = row_count % matrix_length
        X_og_row = row_count % avg_over

        # new_data[0] contains a timestamp we don't need
        X_og[X_og_row, :num_sensors] = new_data[1:]

        # Average the previous readings (if X_og is ready)
        if row_count >= (avg_over-1):
            X[X_row, :] = np.average(X_og, axis=0) #Average of columns
        else:
            X[X_row, :] = new_data

        X[X_row, num_sensors] = new_power
        X_og[X_og_row, num_sensors] = new_power

        print "done getting data"

        # Train the model
        if (row_count % forecasting_interval == 0 and
            (row_count >= matrix_length or init_training)):

            # Log current training windows as pickle files
            with open(Xdata_LOG_FILENAME, 'wb') as logfile:
                pickle.dump(X, logfile)
            with open(Xog_LOG_FILENAME, 'wb') as logfile:
                pickle.dump(X_og, logfile)

            print "training"

            # Unwrap the matrices (put the most recent data on the bottom)
            data = X[X_row:, :num_sensors]
            data = np.concatenate((data, X[:X_row, :num_sensors]), axis=0)
            y = X[X_row:, num_sensors]
            y = np.concatenate((y, X[:X_row, num_sensors]), axis=0)

            # BLR train:
            w_opt, a_opt, b_opt, S_N = train(data, y)

            init_training = True

        # Make a prediction
        if init_training:

            # Prediction is dot product of data and weights
            x_test = X[X_row, :num_sensors]
            prediction = np.inner(w_opt, x_test)
            target = X[X_row, num_sensors]

            # Log the prediction and target results
            logging.info("Target: " + str(target) + "\tPrediction: " + str(prediction))
            y_target.append(target)
            y_predict.append(prediction)
            y_time.append(dt.datetime.fromtimestamp(goal_time).strftime('%Y-%m-%d %H:%M:%S'))

            # Achieve scrolling effect by only graphing most recent data
            if len(y_time) >= matrix_length:
                y_time = y_time[-matrix_length:]
                y_target = y_target[-matrix_length:]
                y_predict = y_predict[-matrix_length:]

            queue.put((y_time, y_target, y_predict)) # app will eventually pick this up

            # Update severity metric and check for anomalies
            error = (prediction-target)
            sigma = np.sqrt(1/b_opt + np.dot(np.transpose(x_n),np.dot(S_N, x_n)))

            # Catch pathogenic cases where variance gets too small
            if sigma < 1:
                sigma = 1

            mu = mu; sigma = sigma
            Sn, Zn = severityMetric(error, mu, sigma, w, Sn_1)

            # Report to user if error is greater than allowed threshold
            # Uses two-in-a-row counter, much like branch prediction
            if np.abs(Sn) <= THRESHOLD:
                alert_counter = 0
            elif np.abs(Sn) > THRESHOLD and alert_counter == 0:
                alert_counter = 1
                Sn = Sn_1
            elif np.abs(Sn) > THRESHOLD and alert_counter == 1:
                Sn = 0
                alert_counter = 0
                logging.warning("ANOMALY DETECTED")

            Sn_1 = Sn

        # Increment and loop
        row_count += 1

    print "Program killed"
