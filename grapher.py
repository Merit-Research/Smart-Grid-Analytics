#!/usr/bin/env python

# Filename:         grapher2.py
# Contributors:     apadin
# Start Date:       2016-06-24

"""Updated version of grapher using PyQt4.

This program graphs files with the following format:

Timestamp,Target,Prediction,Anomaly
1464763755,9530,26466,0

- Timestamp is an integer representing a UTC timestamp
- Target and Prediction are power values in Watts
- Anomaly is a binary value (1 or 0) indicating whether or not
    this target-prediction pair is an anomaly or not.
"""


#==================== LIBRARIES ====================#
import os
import sys
import csv
import time
import datetime as dt
import numpy as np

from PyQt4 import QtGui, QtCore
#from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import LinearLocator
from matplotlib.figure import Figure

from param import *
from algoFunctions import movingAverage


#==================== GUI CLASSES ====================#
class ResultsGraph(FigureCanvas):

    """Figure class used for graphing results"""

    def __init__(self, parent=None, width=5, height=4, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # Create graphs and lines
        self.graph_power = self.fig.add_subplot(211)
        self.graph_error = self.fig.add_subplot(212)
        zero = dt.datetime.fromtimestamp(0)
        one = dt.datetime.fromtimestamp(1)
        x, y = [zero, one], [-1, -1]
        self.predict_line, = self.graph_power.plot(x, y, color='0.8')
        self.target_line, = self.graph_power.plot(x, y, color='r', linestyle='--')
        self.error_line, = self.graph_error.plot(x, y, color='r')
        self.color_spans = []

        # Change settings of graph
        self.graph_power.set_ylabel("Power (kW)")
        self.graph_error.set_ylabel("Error (kW)")
        self.graph_power.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        self.graph_error.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        self.graph_power.xaxis.set_major_locator(LinearLocator(numticks=7))
        self.graph_error.xaxis.set_major_locator(LinearLocator(numticks=7))

        # Rotate dates slightly
        plt.setp(self.graph_power.get_xticklabels(), rotation=10)
        plt.setp(self.graph_error.get_xticklabels(), rotation=10)

        # Let graph expand with window
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        self.fig.tight_layout()
        self.draw()
        
    # Update the graph using the given data
    # 'times' should be datetime objects
    # 'target' should be float values in Watts
    # 'predict' should be float values in Watts
    def graphData(self, times, target, predict):
        assert(len(times) == len(target))
        assert(len(times) == len(predict))

        # Convert to kW and generate error line
        target = [i/1000.0 for i in target]
        predict = [i/1000.0 for i in predict]
        error = [predict[i] - target[i] for i in xrange(len(times))]

        # Determine new bounds of graph
        xmin = min(times)
        xmax = max(times)
        ymin = 0
        ymax = max(max(target), max(predict)) * 1.1
        emin = min(error)
        emax = max(error)
        self.graph_power.set_xlim(xmin, xmax)
        self.graph_power.set_ylim(ymin, ymax)
        self.graph_error.set_xlim(xmin, xmax)
        self.graph_error.set_ylim(emin, emax)

        # Set data to lines and re-draw graph
        self.predict_line.set_data(times, predict)
        self.target_line.set_data(times, target)
        self.error_line.set_data(times, error)
        self.fig.tight_layout()
        self.draw()

    # Add a vertical color span to the target-prediction graph
    # 'start' should be a datetime (preferably in range)
    # 'duration' should be the width of the span in minutes
    # 'color' should be a string describing an _acceptable_ color value
    def colorSpan(self, start, duration, color):
        end = start + dt.timedelta(minutes=duration)
        span = self.graph_power.axvspan(xmin=start, xmax=end, color=color, alpha=0.2)
        self.color_spans.append(span)
        self.fig.tight_layout()
        self.draw()

    # Remove all vertical color spans
    def clearSpans(self):
        for span in self.color_spans:
            span.remove()
        self.color_spans = []
        self.fig.tight_layout()
        self.draw()

        
class PowerGraph(FigureCanvas):

    """Figure class used for graphing"""

    def __init__(self, parent=None, width=5, height=4, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PowerGraph, self).__init__(self.fig)
        
        self.setParent(parent)

        self.graph = self.fig.add_subplot(111)
        self.clear()

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)
        self.fig.tight_layout()
        self.draw()

    # Update the graph using the given data
    # 'times' should be datetime objects
    # 'power' should be in Watts
    def graphData(self, times, power):
        power = [i/1000.0 for i in power]
        xmin = min(times)
        xmax = max(times)
        ymin = 0
        ymax = max(power) * 1.1

        #self.graph.plot(times, power, 'r')
        self.power_line.set_data(times, power)
        self.graph.set_xlim(xmin, xmax)
        self.graph.set_ylim(ymin, ymax)

        self.fig.tight_layout()
        self.draw()
        
    # Add a horizontal color span to the graph
    # 'start' should be a datetime (preferably in range)
    # 'duration' should be the width of the span
    # 'color' should be a string describing an acceptable color value
    def colorSpan(self, start, duration, color):
        end = start + dt.timedelta(minutes=duration)
        self.graph.axvspan(xmin=start, xmax=end, color=color, alpha=0.2)
        self.draw()

    # Clear current graph, including line and 
    def clear(self):
        self.graph.cla()
        zero = dt.datetime.fromtimestamp(0)
        one = dt.datetime.fromtimestamp(1)
        x, y = [zero, one], [-1, -1]
        self.graph.set_xlim(zero, one)
        self.graph.set_ylim(0, 1)
        self.power_line, = self.graph.plot(x, y, color='red')
        self.graph.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        self.graph.xaxis.set_major_locator(LinearLocator(numticks=6))
        plt.setp(self.graph.get_xticklabels(), rotation=10)
        self.graph.set_ylabel("Power (kW)")
        self.fig.tight_layout()
        self.draw()
        
        
class ResultsWindow(QtGui.QMainWindow):

    """Main application window, creates the widgets and the window"""

    # Constructor
    def __init__(self):
        super(ResultsWindow, self).__init__()
        self.setGeometry(50, 50, 1200, 800)
        self.setWindowTitle('Results Grapher')
        self.setWindowIcon(QtGui.QIcon(ICON_FILE))
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Create top-level widget and immediate children
        self.statusBar()
        main_widget = QtGui.QWidget()
        self.graph_widget = self.graphWidget()
        self.settings_widget = self.settingsWidget()

        # Add children to layout and set focus to main widget
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.graph_widget)
        layout.addWidget(self.settings_widget)
        main_widget.setLayout(layout)
        main_widget.setFocus()
        self.setCentralWidget(main_widget)
        self.show()
        
    #==================== WIDGET FUNCTIONS ====================#
    # These functions create all the widgets, sub-widgets, etc. of the program.
    # Each function creates a new widget instance, adds all necessary layouts
    # and sub-widgets, and then returns its widget to the widget above.

    # Create an instance of the ResultsGraph widget
    def graphWidget(self):
        main_widget = QtGui.QWidget(self)
        layout = QtGui.QVBoxLayout()
        self.canvas = ResultsGraph(main_widget, width=5, height=4, dpi=80)
        toolbar = NavigationToolbar(self.canvas, main_widget)

        layout.addWidget(self.canvas)
        layout.addWidget(toolbar)
        main_widget.setLayout(layout)
        return main_widget

    # Create the settings window below the grapher
    def settingsWidget(self):
        main_widget = QtGui.QWidget(self)
        file_widget = self.fileWidget(main_widget)
        self.options_widget = self.optionsWidget(main_widget)
        self.options_widget.setDisabled(True)

        layout = QtGui.QHBoxLayout(main_widget)
        layout.addWidget(file_widget)
        layout.addWidget(self.options_widget)
        main_widget.setLayout(layout)
        return main_widget

    # Creates the filename bar and browse button
    def fileWidget(self, parent):
        main_widget = QtGui.QWidget(parent)
        layout = QtGui.QGridLayout()
        
        file_label = QtGui.QLabel("Results file: ", main_widget)
        file_label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.start_label = QtGui.QLabel(" ", main_widget)
        self.end_label = QtGui.QLabel(" ", main_widget)
        self.file_edit = QtGui.QLineEdit(main_widget)
        browse = QtGui.QPushButton('Browse...')
        browse.clicked.connect(self.browseFile)
        
        layout.addWidget(file_label, 0, 0)
        layout.addWidget(self.file_edit, 1, 0)
        layout.addWidget(browse, 1, 1)
        layout.addWidget(self.start_label, 2, 0)
        layout.addWidget(self.end_label, 3, 0)
        
        main_widget.setLayout(layout)
        return main_widget

    # Creates the options panel to toggle smoothing and anomalies
    def optionsWidget(self, parent):
        main_widget = QtGui.QWidget(parent)
        layout = QtGui.QFormLayout()

        self.options_label = QtGui.QLabel("Options:", main_widget)
        self.options_label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        #self.options_label.setAlignment(QtCore.Qt.AlignCenter)
        self.smooth_box = QtGui.QCheckBox("Smooth data (minutes):    ", main_widget)
        self.smooth_box.stateChanged.connect(self.smoothToggled)
        self.smooth_edit = QtGui.QLineEdit(main_widget)
        self.smooth_edit.setText('0')
        self.smooth_edit.setDisabled(True)
        self.anomaly_box = QtGui.QCheckBox("Show anomalies", main_widget)
        self.anomaly_box.stateChanged.connect(self.anomalyToggled)
        #reset = QtGui.QPushButton("Reset", main_widget)
        #reset.clicked.connect(self.resetOptions)
        update = QtGui.QPushButton("Update Graph", main_widget)
        update.clicked.connect(self.updateGraph)
        
        layout.addRow(self.options_label)
        layout.addRow(self.smooth_box, self.smooth_edit)
        layout.addRow(self.anomaly_box)
        layout.addRow(update)
        
        main_widget.setLayout(layout)
        return main_widget
        
    #==================== HELPER FUNCTIONS ====================#
    # These functions do the actual work of the program.
    # Most are called in response to an event triggered by the main window, while
    # others are helper functions which perform a simple task.

    # Return true if the given filename is valid, false otherwise
    def checkFilename(self, filename):
        if filename == '':
            self.statusBar().showMessage("Error: no file name given")
            return False
        elif filename[-4:] != '.csv':
            self.statusBar().showMessage("Error: file must be '.csv' format")
            return False
        return True
        
    # Open the file search dialog window and get the resulting filename
    def browseFile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName())
        self.file_edit.setText(filename)
        if (filename != ''):
            if (self.anomaly_box.isChecked()):
                self.anomaly_box.toggle()
            if (self.smooth_box.isChecked()):
                self.smooth_box.toggle()
            self.loadFile()
            self.canvas.graphData(self.times, self.target, self.predict)
            self.options_widget.setEnabled(True)
            self.statusBar().showMessage("Graphing complete.", 5000)

    # Load data from the file given by filename
    def loadFile(self):
        filename = str(self.file_edit.text())

        if self.checkFilename(filename):
            try:
                file = open(filename, 'rb')
            except:
                filename = os.path.basename(filename)
                self.statusBar().showMessage(
                    "Error: file %s was not found" % filename)
            else:
                reader = csv.reader(file)
                headers = reader.next()

                self.times = []
                self.target = []
                self.predict = []
                self.anomalies = []
                for row in reader:
                    self.times.append(row[0])
                    self.target.append(float(row[1]))
                    self.predict.append(float(row[2]))
                    try:
                        self.anomalies.append(float(row[3]))
                    except IndexError:
                        pass
                file.close()

                # Convert times from string or timestamp to datetime
                try:
                    self.times = [
                        dt.datetime.fromtimestamp(float(t)) for t in self.times]
                except ValueError:
                    self.times = [
                        dt.datetime.strptime(t, DATE_FORMAT) for t in self.times]

                self.start_label.setText(
                    "Start time: %s" % dt.datetime.strftime(self.times[0], DATE_FORMAT))
                self.end_label.setText(
                    "End time: %s" % dt.datetime.strftime(self.times[-1], DATE_FORMAT))
                
    # Decide what to do based on the state of the smooth checkbox
    def smoothToggled(self, state):
        if state == QtCore.Qt.Checked:
            self.smooth_edit.setEnabled(True)
        else:
            self.smooth_edit.setDisabled(True)
            self.smooth_edit.setText('0')
            
    # Decide what to do based on the state of the anomaly checkbox
    def anomalyToggled(self, state):
        if state == QtCore.Qt.Checked:
            self.showAnomalies()
        else:
            self.canvas.clearSpans()
    
    # Decide what to do based on the state of the smooth checkbox
    def updateGraph(self):
        try:
            smoothing_window = int(self.smooth_edit.text())
        except:
            error_window = QtGui.QErrorMessage(self)
            error_window.showMessage("Smoothing window must be integer value.")
        else:
            if smoothing_window > 0:
                self.canvas.graphData(
                    self.times,
                    movingAverage(self.target, smoothing_window),
                    movingAverage(self.predict, smoothing_window))
            else:
                self.canvas.graphData(self.times, self.target, self.predict)
            self.statusBar().showMessage("Graphing complete.", 5000)
                
    # Reset the settings to default and draw the original graph
    def resetOptions(self):
        self.smooth_box.setCheckState(QtCore.Qt.Unchecked)
        self.anomaly_box.setCheckState(QtCore.Qt.Unchecked)
        self.canvas.clearSpans()
        self.canvas.graphData(self.times, self.target, self.predict)

    # Draw colored bars to show regions where anomalies happened
    def showAnomalies(self):
        self.settings_widget.setDisabled(True)
        loading_win = LoadingWindow()

        count = 0
        anomaly_count = 0
        start = self.times[0]
        dur = 60
        level1 = 0
        level2 = dur / 3.0
        level3 = level2 * 2
        self.canvas.clearSpans() #Clear any existing spans

        for i in self.anomalies:
            anomaly_count += self.anomalies[count]
            if ((count+1) % dur) == 0:
                if anomaly_count >   level3: self.canvas.colorSpan(start, dur, 'red')
                elif anomaly_count > level2: self.canvas.colorSpan(start, dur, 'orange')
                elif anomaly_count > level1: self.canvas.colorSpan(start, dur, 'green')
                anomaly_count = 0
                start = self.times[count]
                QtGui.QApplication.processEvents()
            count += 1
        
        self.settings_widget.setEnabled(True)
        loading_win.close()


class LoadingWindow(QtGui.QDialog):

    """Create a "loading window" which has an infinitely running progress bar"""

    def __init__(self):
        super(LoadingWindow, self).__init__(None, QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle(' ')
        self.setWindowIcon(QtGui.QIcon(ICON_FILE))

        layout = QtGui.QVBoxLayout()
        layout.addWidget(QtGui.QLabel("Calculating. Please wait...", self))
        progress = QtGui.QProgressBar(self)
        progress.setMinimum(0)
        progress.setMaximum(0)
        layout.addWidget(progress)
        self.setLayout(layout)
        self.show()
        

#==================== MAIN ====================#
def main(argv):
    app = QtGui.QApplication(sys.argv)
    toplevel = ResultsWindow()
    sys.exit(app.exec_())


#==================== DRIVER ====================#
if __name__ == '__main__':
    main(sys.argv)
