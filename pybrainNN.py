# Neural Network - Regressor algorithm
# Filename:     pybrainNN.py
# Author:       apadin
# Start Date:   7/26/2016


#==================== LIBRARIES ====================#

import sys
import time
import numpy as np

from matplotlib import pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer


#==================== CLASSES ====================#

class NeuralNetwork(object):
    """Class wrapper for NeuralNetwork functions."""

    def __init__(self, layers):
        """Creates a neural network with shape described by 'layers'."""
        self.network = buildNetwork(*layers, bias=True, hiddenclass=LinearLayer)

    def train(self, X, y, Lambda=0.01):
        """Train the network on the given data."""
        data = SupervisedDataSet(X, y)
        trainer = BackpropTrainer(self.network, data)
        trainer.trainUntilConvergence(maxEpochs=50)
        
    def predict(self, X):
        """Make a prediction based on the most recent training"""
        X = np.ravel(X)
        return float(self.network.activate(X))


def main(argv):

    print "Testing neural network"
    
    m = 1000
    n = 10

    np.random.seed(0)
    X = np.matrix(np.random.sample((m, n)))
    theta = np.matrix(np.random.sample((1, n))) * 10
    y = X * theta.T + 6

    t = time.time()
    neuralNet = NeuralNetwork((n, n/2, 1))
    neuralNet.train(X, y, Lambda=1)
    print "Training completed in: %.3f seconds" % (time.time() - t)
    
    for i in xrange(20):
        X = np.matrix(np.random.sample((1, n)))
        pred1 = float(X * theta.T + 6)
        pred2 = neuralNet.predict(X)
        print "%.3f\t%.3f" % (pred1, pred2)
        
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))







