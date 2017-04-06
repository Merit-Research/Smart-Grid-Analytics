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
import os
import sys
import time
import datetime as dt
import csv
import io
import argparse
import gzip
import shutil
import tarfile
from Queue import Empty as QueueEmpty
from multiprocessing import Process, Queue

import zway


#==================== FUNCTIONS ====================#

def get_all_data(server):
    """Accept a zway.Server object and returns data for all connected devices."""
    return [server.get_data(id) for id in server.device_IDs()]

def unzip_and_archive(prefix, folder):
    """Unzip every file in the folder and then archive and compress the folder."""
    if folder == '': return
    folder = prefix + '/' + folder
    for f in os.listdir(folder):
        if f.endswith('.gz'):
            filename = folder + '/' + f
            try:
                with gzip.open(filename, 'rb') as fin, open(filename, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
            except IOError:
                print filename, "not found"
                pass
    with tarfile.open(folder + ".tar.gz", "w:gz") as tarball:
        tarball.add(folder)
    shutil.rmtree(folder)

def compress(path):
    """Compress the given file to a copy with the same name plus the .gz extension."""
    origfile = path[-1]
    gzipfile = "/".join(path) + '.gz'
    if origfile == '': return
    try:
        os.makedirs(os.path.dirname(gzipfile))
    except OSError:
        pass
    try:
        with open(origfile, 'rb') as fin, gzip.open(gzipfile, 'wb') as fout:
            shutil.copyfileobj(fin, fout)
    except IOError:
        print "Could not compress", origfile, "because it does not exist."
    else:
        os.remove(origfile)
        print "Successfully compressed", gzipfile


#==================== CLASSES ====================#

class Datalog(object):
    """Wrapper for reading and writing sequential data to CSV files"""

    def __init__(self, prefix, header):
        """Specify the prefix of the files you want to write.
        - prefix: The files will have the format "prefix_YYYY-MM-DD.csv"
        - header: List of names of each column (exclude "timestamp")

        Filesystem has the following format:

        prefix (top level)
            folder (month)
                filename (day)
        """
        self.prefix = prefix
        self.folder = ""
        self.fname = ""

        # Generate the header
        header.insert(0, "timestamp")
        output = io.BytesIO()
        csv.writer(output).writerow(header)
        self.header = output.getvalue()

        # Initialize the file-writing process
        self.queue = Queue()
        Process(target=self.file_process).start()

    def log(self, sample, timestamp=None):
        """Add a new sample to the file system.
        - sample: List of values to be written (excluding timestamp)
        """
        if (timestamp == None):
            timestamp = time.time()
        sample.insert(0, timestamp)
        self.queue.put(list(sample))
        
    def file_process(self):
        """Process for handling writing and compressing files."""
        while True:
            try:
                sample = self.queue.get(timeout=5)
            except QueueEmpty:
                pass
            else:
                timestamp = sample[0]
                date = dt.date.fromtimestamp(timestamp)
                fname = "{}_{}.csv".format(self.prefix, date)

                # Check if new file is needed
                if fname != self.fname:
                    # Compress the old file or files and start anew
                    compress([self.prefix, self.folder, self.fname])
                    folder = "{}-{:0>2}".format(date.year, date.month)
                    if folder != self.folder:
                        unzip_and_archive(self.prefix, self.folder)
                        self.folder = folder
                    self.start_new_file(fname)
                    self.fname = fname
                # Record the sample
                with open(fname, 'ab') as fh:
                    csv.writer(fh).writerow(sample)

    def start_new_file(self, fname):
        """Start a new file with the given filename."""
        try:
            with open(fname, 'rb') as fh:
                assert(self.header == fh.readline())
        except AssertionError:
            raise RuntimeError("""existing file with the same name but different settings already exists. Please choose another name for this file""")
        except IOError:
            with open(fname, 'wb') as fh:
                fh.write(self.header)

    def read_range(self, start, end):
        raise RuntimeError("Not yet implemented")


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



