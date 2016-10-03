#!/usr/bin/python

# Grapher class for plotting target and prediction values
# Filename:     grapher.py
# Author(s):    apadin
# Start Date:   5/13/2016

import argparse
import time
import sys
import datetime as dt
import numpy as np
import csv
from multiprocessing import Process, Queue, freeze_support
from Queue import Empty as QueueEmpty

from algoRunFunctions import movingAverage

import Tkinter as Tk    # GUI Library

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from matplotlib.ticker import LinearLocator
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg


##############################  DEFINITIONS  ##############################
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_FILE = 'results.csv'
ICON_FILE = 'app/merit_icon.ppm'


##############################  HELPER FUNCTIONS  ##############################

# Give the window a title and icon, destroy cleanly when X is pressed
def initWindow(window, title=" ", root=False):
    window.wm_title(title)                              # Change title
    icon = Tk.PhotoImage(file = ICON_FILE)                 # Change icon
    window.tk.call('wm', 'iconphoto', window._w, icon)

    def quit_and_destroy(window):
        window.quit()
        window.destroy()
        sys.exit(0)

    window.protocol("WM_DELETE_WINDOW", lambda: quit_and_destroy(window))


##############################  GRAPHFRAME CLASS  ##############################
class GraphFrame(Tk.Frame):

    # Constructor
    def __init__(self, master, queue):

        Tk.Frame.__init__(self, master)
        Tk.Grid.rowconfigure(self, 0, weight=1)
        Tk.Grid.columnconfigure(self, 0, weight=1)

        # Update the graph periodically with new data
        self.graph_queue = queue
        self.after_idle(self.checkQueue)

        # Create figure and add subplots
        self.fig = Figure()
        self.graph_predict = self.fig.add_subplot(211) # Target versus prediction
        self.graph_error = self.fig.add_subplot(212) # Error (target - prediction)

        # Set titles and axis labels for both graphs
        #self.fig.suptitle("Sequential BLR: Prediction and Error", fontsize=18)
        #self.graph_predict.set_title("Prediction vs. Target")
        #self.graph_predict.set_xlabel("Time")
        self.graph_predict.set_ylabel("Power (Watts)")
        #self.graph_error.set_title("Error (Prediction minus Target)")
        #self.graph_error.set_xlabel("Time")
        self.graph_error.set_ylabel("Error (Watts)")

        # Add lines and legend
        x, y = [1, 2], [0, 0]
        self.predict_line, = self.graph_predict.plot(x, y, color='0.75')
        self.target_line, = self.graph_predict.plot(x, y, color='blue', linestyle='--')
        self.error_line, = self.graph_error.plot(x, y, color='red')

        self.graph_predict.legend([self.target_line, self.predict_line], ["Target", "Prediction"])
        self.graph_error.legend([self.error_line], ["Error"])

        # Sets the x-axis to only show hours, minutes, and seconds of time
        self.graph_predict.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M:%S"))
        self.graph_error.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M:%S"))

        # Sets the x-axis to only show 6 tick marks
        self.graph_predict.xaxis.set_major_locator(LinearLocator(numticks=6))
        self.graph_error.xaxis.set_major_locator(LinearLocator(numticks=6))

        # Angle the labels slightly so they are more distinguishable
        labels = self.graph_predict.get_xticklabels()
        plt.setp(labels, rotation=10)
        labels = self.graph_error.get_xticklabels()
        plt.setp(labels, rotation=10)

        # Tk canvas and toolbar which are embedded into application
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side='top', fill='both', expand=True)


    # Check the queue for new data and graph if exists
    def checkQueue(self):
        
        # If there is data, graph it. If not, whatever
        try:
            contents = self.graph_queue.get(block=False)
        except QueueEmpty:
            pass
        else:
                
            # Check if this is graph info or anomaly info
            if len(contents) == 4:
                if   contents[0] == 'a': self.graph_anomalies(*(contents[1:]))
                elif contents[0] == 'd': self.graph(*(contents[1:]))

        # Always schedule another check in a reasonable amount of time
        # DO NOT change to after_idle! Program will freeze
        self.after(200, self.checkQueue)


    # Anomaly stuff
    def graph_anomalies(self, start, end, count):

        # Time should be UNIX timestamp. Convert to datetime
        start = dt.datetime.fromtimestamp(start)
        end = dt.datetime.fromtimestamp(end)
        
        if   count > 0  and count <= 10: color = 'green'
        elif count > 10 and count <= 20: color = 'orange'
        elif count > 20 and count <= 30: color = 'red'
        else: return

        self.graph_predict.axvspan(xmin=start, xmax=end, color=color, alpha=0.3)


    # Plot the data
    def graph(self, y_time, y_target, y_predict):

        # Time could be datetime string or UNIX timestamp
        try:
            y_time = [dt.datetime.fromtimestamp(float(t)) for t in y_time]
        except ValueError:
            y_time = [dt.datetime.strptime(t, DATE_FORMAT) for t in y_time]

    
        # Time should be UNIX timestamp. Convert to datetime
        #y_time = [dt.datetime.fromtimestamp(t) for t in y_time]

        # Calculate the error vector
        y_error = []
        for i in xrange(len(y_target)):
            y_error.append(y_predict[i] - y_target[i])

        # Set x and y axis limits
        # Axes update every time to achieve "scrolling" effect
        xmin = min(y_time)
        xmax = max(y_time)

        ymin = 0.0
        #ymin = min(min(y_target), min(y_predict))
        ymax = max(max(y_target), max(y_predict))

        emin = min(y_error)
        emax = max(y_error)

        self.graph_predict.set_xlim(xmin, xmax)
        self.graph_predict.set_ylim(ymin, ymax)

        self.graph_error.set_xlim(xmin, xmax)
        self.graph_error.set_ylim(emin, emax)
        #self.graph_error.set_ylim(-1000, 1000)

        # Set new data and graph
        self.predict_line.set_data(y_time, y_predict)
        self.target_line.set_data(y_time, y_target)
        self.error_line.set_data(y_time, y_error)

        labels = self.graph_predict.get_xticklabels()
        plt.setp(labels, rotation=10)
        labels = self.graph_error.get_xticklabels()
        plt.setp(labels, rotation=10)

        self.fig.tight_layout()
        self.canvas.show()

        
##############################  GRAPHER CLASS  ##############################
class Grapher:

    # Create the transfer queue and start the graph process
    def __init__(self):
        self.graph_queue = Queue() # Queue for transferring graph data
        self.graph_proc = Process(target=self.start)
        self.graph_proc.start() #, args=(self.graph_queue,)).start()

    # Open the new grapher window and run it
    def start(self):
        root = Tk.Tk()
        initWindow(root, "Results Graph")
        graph_frame = GraphFrame(master=root, queue=self.graph_queue)
        graph_frame.pack(side='right', fill='both', expand=True)
        Tk.mainloop()

    # Add anomaly rerort to the graph queue
    def graph_anomalies(self, start, end, count):
        self.graph_queue.put(('a', start, end, count))

    # Add data to the graph queue for the grapher to pick up
    def graph(self, y_time, y_target, y_predict):
        self.graph_queue.put(('d', y_time, y_target, y_predict))

    # Delete the process at the end of the program (if you want)
    def close(self):
        self.graph_proc.terminate()


##############################  CSV FUNCTIONS  ##############################

# Read in a results file and return the three data lists
def readResults(csvfile):
    with open(csvfile, 'rb') as infile:
        reader = csv.reader(infile)
        reader.next() #Skip header row
        y_time, y_target, y_predict = [], [], []
        
        for row in reader:
            
            # Make sure none of the entries was corrupted
            #try:    y_time.append(float(row[0]))
            #except: break
            y_time.append(row[0])

            try:    y_target.append(float(row[1]))
            except: y_time.pop(); break
            
            try:    y_predict.append(float(row[2]))
            except: y_time.pop(); y_target.pop(); break

    return y_time, y_target, y_predict


# Save results to a file for later graphing
# 'csvfile' is the name of the file to be written to
# 'results' is a tuple of iterables of the same length with the data to write
def writeResults(csvfile, results):
    with open(csvfile, 'wb') as outfile:
        writer = csv.writer(outfile)
        row_list = zip(*results) #Get columns out of rows
        for row in row_list:
            writer.writerow(row)
            

##############################  CSV CLASS  ##############################
class CSV:

    # Constructor
    def __init__(self, datafile = DEFAULT_FILE):
        self.datafile = datafile


    # Reset the CSV and write the header
    # Deletes all previous data in the file
    def clear(self):
        with open(self.datafile, 'wb') as outfile:
            outfile.write('Time,Target,Prediction\n') # Write the header

    # Append given data to the CSV file
    def append(self, y_time, y_target, y_predict):

        file = open(self.datafile, 'ab')

        assert(len(y_time) == len(y_target))
        assert(len(y_time) == len(y_predict))

        # y_time should be a list of UNIX timestamps
        y_time = [dt.datetime.fromtimestamp(float(t)).strftime(DATE_FORMAT) for t in y_time]
        y_target = [str(t) for t in y_target]
        y_predict = [str(t) for t in y_predict]

        for i in xrange(len(y_time)):
            file.write(y_time[i] + ',' +  y_target[i] + ',' + y_predict[i] + '\n')
        file.close()


    # Read the data in the CSV file and return results
    # Target and prediction are floats, time contains strings
    def read(self):

        file = open(self.datafile, "rb")
        file.next() # Throw out the header row

        y_time, y_target, y_predict = [], [], []

        for line in file:
            line = line.rstrip() #Remove newline
            data = line.split(',')

            # Only grow list if CSV was written properly
            if len(data) == 3:

                # Could be a timestamp or a datetime string
                try:
                    y_time.append(float(data[0]))
                except ValueError:
                    y_time.append(data[0])

                y_target.append(float(data[1]))
                y_predict.append(float(data[2]))

        file.close()
        
        

        return y_time, y_target, y_predict


        
##############################  STATISTICS  ##############################

# Prediction Mean Squared Error
def print_stats(y_target, y_predict, smoothing_win=120):

    T = len(y_target)
    y_target = np.asarray(y_target)
    y_predict = np.asarray(y_predict)

    try:
        y_target_smoothed = movingAverage(y_target, smoothing_win)
        y_predict_smoothed = movingAverage(y_predict, smoothing_win)
    except ValueError as e:
        print repr(e)
        print "Error: Smoothing window cannot be larger than number of data points"
        y_target_smoothed = movingAverage(y_target, 1)
        y_predict_smoothed = movingAverage(y_predict, 1)

    # Prediction Mean Squared Error (smooth values)
    PMSE_score_smoothed = np.linalg.norm(y_target_smoothed-y_predict_smoothed)**2 / T
    # Prediction Mean Squared Error (raw values)
    PMSE_score = np.linalg.norm(y_target - y_predict)**2 / T
    # Relative Squared Error
    Re_MSE = np.linalg.norm(y_target-y_predict)**2 / np.linalg.norm(y_target)**2
    # Standardise Mean Squared Error
    SMSE =  np.linalg.norm(y_target-y_predict)**2 / T / np.var(y_target)

    print "---------------------------------------------------------------------------"
    print "%20s |%20s |%15s |%10s "  % ("RMSE-score (smoothed)", "RMSE-score (raw)", "Relative MSE", "SMSE")
    print "%20.2f  |%20.2f |%15.2f |%10.2f " % (np.sqrt(PMSE_score_smoothed), np.sqrt(PMSE_score), Re_MSE, SMSE)
    print "---------------------------------------------------------------------------"


##############################  MAIN  ##############################
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Graph data from a file using matplotlib tools.')

    parser.add_argument('-n', '--nograph', action='store_true',
                        help='show only statistics, no graph')
    parser.add_argument('-r', '--realtime', nargs='?',metavar = 'TIME', const=5, type=int,
                        help='update the graph in realtime every TIME seconds (default 5)')
    parser.add_argument('-s', '--smooth', nargs='?', metavar='WINDOW', const=120, type=int,
                        help='smooth data with a smoothing window of WINDOW (default 120)')
    parser.add_argument('-f', '--file', metavar='FILE', type=str,
                        help='specify which file to read data from')

    args = parser.parse_args()


    # Get the filename if -f was set
    if args.file != None: filename = args.file
    else:                 filename = 'results.csv'

    # If -n set, show statistics and then exit cleanly
    if args.nograph:
        y_time, y_target, y_predict = readResults(filename)
        print "\nStatistics:"
        print_stats(y_target, y_predict)
        exit(0)

    # Otherwise, create the grapher
    grapher = Grapher()

    # Get the period if -r was set
    try:    period = float(args.realtime)
    except: period = 0

    # Get the smoothing rate if -s was set
    try:    smooth = int(args.smooth)
    except: smooth = 0

    # If -r was set, this loop will continuously graph the data
    # in the given filename every T seconds. If not, the loop
    # will run only once and the program will close when the window
    # closes.
    while True:

        y_time, y_target, y_predict = readResults(filename)

        current_time = y_time[-1]
        print "\nAt time %s" % str(current_time)
        
        print_stats(y_target, y_predict)

        # Smooth data if requested
        if smooth > 0:
            y_target = movingAverage(y_target, args.smooth)
            y_predict = movingAverage(y_predict, args.smooth)
            
        grapher.graph(y_time, y_target, y_predict)

        # If running realtime, reschedule another update
        if period > 0.0:
            time.sleep(period)
        else:
            print "\nClose window to exit"
            sys.exit(0)

# If run as main:
if __name__ == "__main__":
    main()
