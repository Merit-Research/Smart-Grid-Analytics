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
import pandas as pd
#from multiprocessing import Process, Queue

from PyQt4 import QtGui, QtCore
#from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import LinearLocator
from matplotlib.figure import Figure

from modules.common import *
from modules.stats import moving_average


#==================== GUI CLASSES ====================#
class ResultsGraph(FigureCanvas):

    """Figure class used for graphing results"""

    def __init__(self, parent=None, width=5, height=4, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

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

        #self.updateGeometry()
        self.fig.tight_layout()
        #self.draw()
        
    # Update the graph using the given data
    # 'times' should be datetime objects
    # 'target' should be float values in Watts
    # 'predict' should be float values in Watts
    def graph_data(self, times, target, predict):
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
    def color_span(self, start, duration, color):
        end = start + dt.timedelta(minutes=duration)
        span = self.graph_power.axvspan(xmin=start, xmax=end, color=color, alpha=0.2)
        self.color_spans.append(span)
        self.fig.tight_layout()

    # Remove all vertical color spans
    def clear_spans(self):
        for span in self.color_spans:
            span.remove()
        self.color_spans = []
        self.fig.tight_layout()
        self.draw()

class FeatureGraph(FigureCanvas):

    """Figure class used for graphing"""

    def __init__(self, parent=None, width=5, height=4, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

        self.graph = self.fig.add_subplot(111)
        self.graph.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        self.graph.xaxis.set_major_locator(LinearLocator(numticks=6))
        plt.setp(self.graph.get_xticklabels(), rotation=10)

        zero = dt.datetime.fromtimestamp(0)
        one = dt.datetime.fromtimestamp(1)
        x, y = [zero, one], [-1, -1]
        self.data_line, = self.graph.plot(x, y)
        self.fig.tight_layout()
        self.draw()

    def graph_data(self, datetimes, values):
        """Graph the given feature data."""
        xmin, xmax = datetimes[0], datetimes[-1]
        ymin, ymax = min(values) * 1.1, max(values) * 1.1
        self.data_line.set_data(times, values)
        self.graph.set_xlim(xmin, xmax)
        self.graph.set_ylim(ymin, ymax)
        self.fig.tight_layout()
        self.draw()

    def clear(self):
        """Clear the existing line"""
        self.graph.cla()
        zero = dt.datetime.fromtimestamp(0)
        one = dt.datetime.fromtimestamp(1)
        x, y = [zero, one], [-1, -1]
        self.graph_data(x, y)
        FigureCanvas.updateGeometry(self)
        self.fig.tight_layout()
        self.draw()
        
class ResultsWindow(QtGui.QMainWindow):

    """Main application window, creates the widgets and the window"""

    def __init__(self):
        """Create the top-level window and tabs"""
        super(ResultsWindow, self).__init__()
        self.setGeometry(50, 50, 1200, 800)
        self.setWindowTitle("Grapher")
        self.setWindowIcon(QtGui.QIcon(ICON_FILE))
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.statusBar()

        # Create tabs
        self.tabs = QtGui.QTabWidget()
        self.init_results_tab()
        self.init_data_tab()
        self.setCentralWidget(self.tabs)
        self.show()
    
    #==================== INITIALIZE TABS ====================#
    def init_results_tab(self):
        """Create the widgets in the Results tab"""
        self.results_tab = QtGui.QWidget()
        self.tabs.addTab(self.results_tab, QtCore.QString("Results"))
        
        # Create immediate children and add to layout
        graph_widget = self.init_results_graph()
        self.results_settings = self.init_results_settings()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.results_settings)
        layout.addWidget(graph_widget)
        self.results_tab.setLayout(layout)

    def init_data_tab(self):
        """Create the widgets in the Data tab"""
        self.data_tab = QtGui.QWidget()
        self.tabs.addTab(self.data_tab, QtCore.QString("Data"))

        # Create immediate children and add to layout
        graph_widget = self.init_data_graph()
        self.data_settings = self.init_data_settings()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.data_settings)
        layout.addWidget(graph_widget)
        self.results_tab.setLayout(layout)
        
    #==================== INTIALIZE WIDGETS ====================#
    
    def init_results_graph(self):
        main_widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        self.canvas = ResultsGraph(main_widget, width=5, height=4, dpi=80)
        toolbar = NavigationToolbar(self.canvas, main_widget)
        layout.addWidget(self.canvas)
        layout.addWidget(toolbar)
        main_widget.setLayout(layout)
        return main_widget

    def init_results_settings(self):
        main_widget = QtGui.QWidget(self)
        results_browse = self.init_results_browse(main_widget)
        self.options_widget = self.init_results_options(main_widget)
        self.options_widget.setDisabled(True)
        layout = QtGui.QHBoxLayout(main_widget)
        layout.addWidget(results_browse)
        layout.addWidget(self.options_widget)
        main_widget.setLayout(layout)
        return main_widget

    def init_results_browse(self, parent):
        main_widget = QtGui.QWidget(parent)
        layout = QtGui.QGridLayout()
        file_label = QtGui.QLabel("Results file: ", main_widget)
        file_label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.start_label = QtGui.QLabel(" ", main_widget)
        self.end_label = QtGui.QLabel(" ", main_widget)
        self.file_edit = QtGui.QLineEdit(main_widget)
        browse = QtGui.QPushButton('Browse...')
        browse.clicked.connect(self.browse_results_file)
        layout.addWidget(file_label, 0, 0)
        layout.addWidget(self.file_edit, 1, 0)
        layout.addWidget(browse, 1, 1)
        layout.addWidget(self.start_label, 2, 0)
        layout.addWidget(self.end_label, 3, 0)
        main_widget.setLayout(layout)
        return main_widget

    def init_results_options(self, parent):
        main_widget = QtGui.QWidget(parent)
        layout = QtGui.QFormLayout()
        options_label = QtGui.QLabel("Options:", main_widget)
        options_label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        smooth_label = QtGui.QLabel("Smooth data (minutes) (0 for none)", main_widget)
        self.smooth_spin = QtGui.QSpinBox(main_widget)
        self.smooth_spin.setValue(0)
        self.anomaly_button = QtGui.QPushButton("Show Anomalies", main_widget)
        self.anomaly_button.clicked.connect(self.anomaly_clicked)
        self.anomaly_state = 0
        update = QtGui.QPushButton("Update Graph", main_widget)
        update.clicked.connect(self.update_graph)
        layout.addRow(options_label)
        layout.addRow(smooth_label, self.smooth_spin)
        layout.addRow(self.anomaly_button)
        layout.addRow(update)
        main_widget.setLayout(layout)
        return main_widget
        
    def init_data_graph(self):
        main_widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        self.data_canvas = FeatureGraph(main_widget, width=5, height=4, dpi=80)
        toolbar = NavigationToolbar(self.data_canvas, main_widget)
        layout.addWidget(self.data_canvas)
        layout.addWidget(toolbar)
        main_widget.setLayout(layout)
        return main_widget
        
    #==================== HELPER FUNCTIONS ====================#

    def check_filename(self, filename):
        """Return true if the given filename is valid, false otherwise"""
        if filename == '':
            self.statusBar().showMessage("Error: no file name given")
            return False
        elif filename[-4:] != '.csv':
            self.statusBar().showMessage("Error: file must be '.csv' format")
            return False
        return True
        
    def browse_results_file(self):
        """Open the file browse dialog and get the filename."""
        filename = str(QtGui.QFileDialog.getOpenFileName())
        self.file_edit.setText(filename)
        if (filename != ''):
            loading_win = LoadingWindow()
            loading_win.setFocus()
            QtGui.QApplication.processEvents()
            self.load_results_file()
            QtGui.QApplication.processEvents()
            self.canvas.graph_data(self.datetimes,
                                   self.results_data[:, 1], 
                                   self.results_data[:, 2])
            loading_win.close()
            self.options_widget.setEnabled(True)
            self.statusBar().showMessage("Graphing complete.", 5000)

    def load_results_file(self):
        """Load data from the file given by filename."""
        filename = str(self.file_edit.text())
        if self.check_filename(filename):
            try:
                self.results_data = pd.read_csv(filename).values
            except IOError:
                message = "Error: file {} was not found".format(filename)
                self.statusBar().showMessage(message)
            self.datetimes = [dt.datetime.fromtimestamp(t) for t in self.results_data[:, 0]]
            self.start_label.setText("Start time: {}".format(self.datetimes[0]))
            self.end_label.setText("End time:   {}".format(self.datetimes[-1]))
            self.smooth_spin.setMaximum(len(self.datetimes))
            
    def anomaly_clicked(self):
        """Turn on or off anomalies"""
        if self.anomaly_button.text() == 'Show Anomalies':
            self.anomaly_button.setText('Hide Anomalies')
            self.show_anomalies()
        else:
            self.anomaly_button.setText('Show Anomalies')
            self.canvas.clear_spans()
    
    def update_graph(self):
        """Update the graph based on new data"""
        smoothing_window = self.smooth_spin.value()
        if smoothing_window > 0:
            self.canvas.graph_data(self.datetimes,
                moving_average(self.results_data[:, 1], smoothing_window),
                moving_average(self.results_data[:, 2], smoothing_window))
        else:
            self.canvas.graph_data(self.datetimes,
                self.results_data[:, 1], 
                self.results_data[:, 2])
            
    def show_anomalies(self):
        """Draw colored bars to show regions where anomalies happened"""
        self.results_settings.setDisabled(True)
        loading_win = LoadingWindow()
        self.canvas.clear_spans() #Clear any existing spans
        start = self.datetimes[0]
        dur = 60
        level1 = 0
        level2 = dur / 3.0
        level3 = level2 * 2
        anomaly_count = 0
        for count in range(len(self.datetimes)):
            anomaly_count += self.results_data[count, 3]
            if ((count+1) % dur) == 0:
                if anomaly_count > 0:
                    if   anomaly_count > level3: self.canvas.color_span(start, dur, 'red')
                    elif anomaly_count > level2: self.canvas.color_span(start, dur, 'orange')
                    elif anomaly_count > level1: self.canvas.color_span(start, dur, 'green')
                    anomaly_count = 0
                start = self.datetimes[count]
                QtGui.QApplication.processEvents()
        self.canvas.draw()
        self.results_settings.setEnabled(True)
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
