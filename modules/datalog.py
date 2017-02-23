#!/usr/bin/env python
# Filename:     datalog.py
# Authors:      apadin, yabskbd, mjmor, dvorva
# Start Date:   5/9/2016

"""
Driver for collecting data from ZWay server and saving it to given
location. This program also maintains a separate log file for providing
device information so that the data can be analyzed later

- Adrian Padin, 1/20/2017
"""

#==================== LIBRARIES ====================#
import sys
import time
import datetime as dt
import csv
import io
import argparse

import zway


#==================== FUNCTIONS ====================#

def get_all_data(server):
    """
    Accepts a zway.Server object and returns data for all connected devices.
    """
    return [server.get_data(id) for id in server.device_IDs()]


#==================== CLASSES ====================#
    
class Datalog(object):
    """Wrapper for reading and writing sequential data to CSV files"""
    
    def __init__(self, prefix, header):
        """
        Specify the prefix of the files you want to write.
        - prefix: The files will have the format "prefix_YYYY-MM-DD.csv"
        - header: List of names of each column (exclude "timestamp")
        """
        self.prefix = prefix
        self.last_fname = None
        
        # Generate the header
        header.insert(0, "timestamp")
        output = io.BytesIO()
        csv.writer(output).writerow(header)
        self.header = output.getvalue()
        
    def log(self, sample, timestamp=None):
        """
        Add a new sample to the correct file.
        - sample: List of values to be written (excluding timestamp)
        """
        if (timestamp == None):
            timestamp = time.time()
        
        sample = list(sample)
        date = dt.date.fromtimestamp(timestamp)
        fname = "{}_{}.csv".format(self.prefix, date)

        # If the file does not exist, make a new one
        try:
            with open(fname, 'rb') as fh:
                assert(self.header == fh.readline())
        except IOError:
            with open(fname, 'wb') as fh:
                fh.write(self.header)
        except AssertionError:
            raise RuntimeError("""existing file with the same name but different settings already exists. Please choose another name for this file""")

        # Record the sample
        sample.insert(0, timestamp)
        with open(fname, 'ab') as fh:
            csv.writer(fh).writerow(sample)
                
    def read_range(self, start, end):
        pass
        
        
def main(argv):
    """Connect to server and start the logging process."""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str)
    parser.add_argument('port', type=str)
    parser.add_argument('prefix', type=str)
    parser.add_argument('-a', '--auth', nargs=2, type=str, default=(None, None))
    args = parser.parse_args(argv[1:])
    
    try:
        username, password = args.auth
    except Exception:
        server = zway.Server(args.host, args.port)
    else:
        server = zway.Server(args.host, args.port, username=username, password=password)


    device_list = server.device_IDs()
    log = Datalog(args.prefix, device_list)
    
    # Timing procedure
    granularity = 10
    goal_time = int(time.time())

    while(True):
        
        while goal_time > time.time():
            time.sleep(0.2)
        goal_time = goal_time + granularity
        print "sample at time", dt.datetime.fromtimestamp(goal_time)
        
        log.log(get_all_data(server), goal_time)

if __name__ == '__main__':
    main(sys.argv)
    


