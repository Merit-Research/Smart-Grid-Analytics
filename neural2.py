# Describes the classes and functions required to run
# a neural network algorithm
# Filename:     neuralNetwork2.py
# Author:       apadin
# Start Date:   8/1/2016


#==================== LIBRARIES ====================#

import sys
import numpy as np

from scipy.optimize import minimize
from matplotlib import pyplot as plt


#==================== FUNCTIONS ====================#

def sigmoid(Z):
    """Calculate element-wise sigmoid function of matrix X."""
    return 1.0 / (1.0 + np.exp(-1.0 * Z))

def sigmoidGradient(Z):
    """Calculate element-wise sigmoid gradient function of matrix X."""
    return np.multiply(sigmoid(Z), (1-sigmoid(Z)));


#==================== CLASSES ====================#

class NeuralNetwork(object):

    """Class wrapper for creating and simulating nueral network algorithms."""

    def __init__(self, layers):
        """Create all necessary structures for training and running the net.
        
        Inputs:
        - layers: size and shape of all layers, including input and output
        - func: callable activation function which takes as an input a matrix
            of any size and returns a matrix of the same size
        - grad: callable function which is the gradient of the activation func
        
        Layers must have the following format:
        - layers[0] is number of features
        - layers[1:-1] is number and size of hidden layers/units
        - layers[-1] is the number of output classes
        
        Therefore, 'layers' must have at least two entries: the number of features
        and the number of output classes. Note that this implies that if 'layers'
        has only two entries, the neural network will have zero hidden layers.
        """
        
        try:
            assert(len(layers) >= 2)
        except AssertionError:
            raise RuntimeError("'layers' must contain at least two elements")
            
        self.num_features = layers[0]
        self.num_classes = layers[-1]
            
        # Create the model parameter matrices
        self.Theta = []
        for i in xrange(len(layers) - 1):
            layer_size = (layers[i+1], layers[i] + 1)
            self.Theta.append(np.matrix(np.zeros(layer_size)))
        
        
        #==================== USER FUNCTIONS ====================#
        
        def train(self, X, y):
            """Train the neural network on the given data."""
            
            # But first, error checking!
            Xm, Xn = X.shape
            ym, yn = y.shape
            assert(Xm == ym)
            assert(Xn == self.num_features)
            assert(yn == self.num_classes)
            
            # Feature Scaling
            self.meanX = np.mean(X, 0)
            self.stdX = np.max(X, 0) - np.min(X, 0)
            X = (X - self.meanX) / self.stdX            

            # Add column of ones
            ones = np.ones((X.shape[0], 1))     # Add column of ones
            X = np.concatenate((ones, X), 1)
            
            # Prepare for minimizing
            epsilon = 0.1
            Theta = np.random.sample(((n+1)**2)) * 2 * epsilon - epsilon
            costFunc = lambda(t): self.costFunc(X, y, Theta, layers)
            
            Theta 
            
            result = minimize(costFunc, Theta, jac=True)
            Theta_opt = result.x


            
        def predict(self, X):
            """Make a prediction based on the most recent training session."""
            
            # But first, error checking!
            Xm, Xn = X.shape
            assert(Xm == ym)
            assert(Xn == self.num_features)
            assert(yn == self.num_classes)

            prediction = self.__forwardprop(X)
        
        
        
        #==================== HELPER FUNCTIONS ====================#








        
        