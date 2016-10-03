# Neural Network - Regressor algorithm
# Filename:     neuralNetwork.py
# Author:       apadin
# Start Date:   7/26/2016


#==================== LIBRARIES ====================#

import sys
import numpy as np

from scipy.optimize import minimize
from matplotlib import pyplot as plt


#==================== FUNCTIONS ====================#

def sigmoid(Z):
    """Calculate element-wise sigmoid function of X.
       X must be a float, vector, or matrix."""
    return 1.0 / (1.0 + np.exp(-1.0 * Z))


def sigmoidPrime(Z):
    """Calculate element-wise sigmoid function of X.
       X must be a float, vector, or matrix."""
    return np.multiply(sigmoid(Z), (1-sigmoid(Z)));


#==================== CLASSES ====================#

class NeuralNetwork(object):

    """Class wrapper for NeuralNetwork functions."""
    def __init__(self, layers):
        """Creates a neural network with shape described by 'layers'."""
        self.layers = layers

    def train(self, X, y, Lambda=0.01):
        """Train the network on the given data."""

        # Feature Scaling
        self.meanX = np.mean(X, 0)
        self.stdX = np.max(X, 0) - np.min(X, 0)
        X = (X - self.meanX) / self.stdX

        # Minimize to find optimum Theta
        Theta = self.__randTheta()
        maxiter = 50
        costFunc = lambda(t): self.__costFunction(t, X, y, Lambda)
        result = minimize(costFunc, Theta, jac=True)
        self.Theta = result.x
        #print result.message


    def predict(self, X):
        """Make a prediction based on the most recent training"""
        X = (X - self.meanX) / self.stdX    # Feature scaling
        prediction = self.__costFunction(self.Theta, X, [], 0, pred_only=True)

        # If the result is a matrix of size larger than 1x1, return the whole
        # matrix. Otherwise only return the float value
        try:
            return float(prediction)
        except TypeError:
            return prediction.T


    def __costFunction(self, Theta, X, y, Lambda, pred_only=False):
        """Return the cost and graident for a given Theta.
        Can also make a prediction if pred_only is set to true."""

        # Setup some useful parameters
        m, n = np.shape(X)
        X = X.T
        
        if not pred_only:
            y = y.T
            
        Theta = self.__unravel(Theta)   # List of 2D arrays
        layers = self.layers            # Shape of network
        num_layers = len(layers)
        num_classes = 1

        cost = 0
        Theta_grad = Theta[:]
        
        # Step 1: Forward propogation
        A = [None] * num_layers
        A[0] = np.concatenate((np.ones((1, m)), X), 0)
        
        for i in xrange(num_layers-1):
            if i == num_layers-2:
                A[i+1] = Theta[i] * A[i]
            else:
                A[i+1] = np.concatenate((np.ones((1, m)), Theta[i] * A[i]), 0)
        
        if pred_only:
            return A[-1]

        # Step 2: Cost function
        cost = np.sum(np.square(A[-1] - y))
        for i in xrange(1, num_layers-1):
            cost += np.sum(np.square(Theta[i])) #Regularization
        
        # Step 3: Backpropogation
        Delta = [None]*num_layers
        Delta[-1] = A[-1] - y
        for i in xrange(num_layers-2, 0, -1):
            Delta[i] = (Theta[i].T * Delta[i+1])[1:, :]
            
        # Step 4: Gradient calculation
        for i in range(num_layers-1):
            Theta_grad[i] = (Delta[i+1] * A[i].T) / m
            Theta_grad[i][:, 1:] = Theta_grad[i][:, 1:] + (Lambda / m) * Theta_grad[i][:, 1:]

        # Step 5: Convert gradient to 1D array
        Theta_grad = self.__ravel(Theta_grad)
        return cost, Theta_grad
            
            
    def __randTheta(self):
        """Generate a list of small random numbers used to initialize Theta."""
        epsilon = 0.02
        layers = self.layers

        num_params = 0;
        for i in range(len(layers) - 1):
            num_params += (layers[i + 1] * (layers[i] + 1))

        return (np.random.rand(num_params) * 2.0 * epsilon) - epsilon
            
            
    '''
    def __gradCheck(self, costFunc, Theta):
        """Return the gradient of the cost for a given Theta.
           This function calculates the gradient numerically."""
        gradient = np.zeros(np.shape(Theta))
        perturb = np.zeros(np.shape(Theta))
        epsilon = 10**-4

        for p in xrange(len(Theta)):
            perturb[p] = epsilon
            loss1, _ = costFunc(Theta - perturb)
            loss2, _ = costFunc(Theta + perturb)
            gradient[p] = (loss2 - loss1) / (2.0*epsilon)
            perturb[p] = 0;

        return gradient
    '''
        
    def __ravel(self, Theta):
        """Convert Theta into a contiguous flat array."""
        Theta_ravel = np.ravel(Theta[0])
        for i in xrange(1, len(Theta)):
            Theta_ravel = np.concatenate((Theta_ravel, np.ravel(Theta[i])))
        return Theta_ravel


    def __unravel(self, Theta):
        """Convert Theta (flat array) back into Theta1 and Theta2."""
        layers = self.layers
        Stack = []
        first = 0
        for i in range(len(layers) - 1):
            shape = [layers[i + 1], (layers[i] + 1)]
            last = first + (shape[0] * shape[1])
            Stack.append(np.reshape(Theta[first:last], shape))
            first = last
        return Stack


def main(argv):

    "Testing neural network"

    np.random.seed(0)
    data = np.matrix(np.random.sample((80, 5)))
    theta = np.matrix([[1, 2, 3, 4, 5]])
    y = data * theta.T + 6

    neuralNet = NeuralNetwork((5, 4, 1))
    neuralNet.train(data, y, Lambda=1)

    X1 = np.matrix(np.random.sample((20, 5)))
    X1 = np.sort(X1)
    y1 = neuralNet.predict(X1)

    #plt.plot(X1[:, 0], y1)
    #plt.plot(X1[:, 1], y1)
    #plt.plot(X1[:, 2], y1)
    #plt.plot(X1[:, 3], y1)
    #plt.plot(X1[:, 4], y1)
    #plt.show()

    print neuralNet.predict([1, 1, 1, 1, 1])

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))



















