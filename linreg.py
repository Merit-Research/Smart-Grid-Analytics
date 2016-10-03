# Classes for linear and polynomial regression
# Filename:     linreg.py
# Author:       apadin
# Start Date:   7/26/2016


#==================== LIBRARIES ====================#

import sys
import numpy as np

from sklearn import linear_model
from scipy.optimize import minimize
from matplotlib import pyplot as plt


#==================== CLASSES ====================#

class SklearnLinReg(object):

    def __init__(self, regparam=0.1):
        self.model = linear_model.Ridge(regparam)
        
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        X = X.reshape(1, -1)
        return float(self.model.predict(X))
    
    
class SklearnBLR(object):

    def __init__(self):
        self.model = linear_model.BayesianRidge()
        
    def train(self, X, y):
        self.model.fit(X, np.ravel(y))

    def predict(self, X):
        X = X.reshape(1, -1)
        return float(self.model.predict(X))


class LinearRegression(object):

    """Regression model based on simple linear regression."""
    
    def __init__(self, Lambda=1):
        self.Lambda = Lambda
        print Lambda

    def train(self, X, y):
        """Train the linear model using the given features and labels."""
        
        # Feature Scaling
        self.meanX = np.mean(X, 0)
        self.stdX = np.std(X, 0)
        X = np.divide((X - self.meanX), self.stdX)

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), 1)
        n = X.shape[1]

        # If not doing regularization, use the normal equation
        if self.Lambda == 0:
            self.theta = np.linalg.pinv(X.T * X) * X.T * y
            
        # Otherwise, use gradient descent (or other optimizer)
        else:
            init_theta = (np.zeros([n, 1]))
            costFunc = lambda(theta): self.__costFunction(X, y, theta, self.Lambda)

            result = minimize(costFunc, init_theta, jac=True)
            self.theta = np.reshape(result.x, [n, 1])

    def predict(self, X):
        """Make a prediction based on the most recent training session."""
        X = np.matrix(X)
        X = (X - self.meanX) / self.stdX    # Feature scaling
        ones = np.ones((X.shape[0], 1))     # Add bias feature
        X = np.concatenate((ones, X), 1)
        return float(X * self.theta)
        
    def __costFunction(self, X, y, theta, Lambda):
        """Calculate the cost function and gradient for a given
           value of theta. Also requires a regularization parameter
           Lambda."""
        
        m, n = np.shape(X)
        shape = np.shape(theta)
        theta = np.reshape(theta, [n, 1])
        
        # Cost function is defined as sum of the square error
        cost = (1.0 / (2.0*m)) * np.sum(np.square(X*theta - y))
        cost += (Lambda / (2*m)) * np.sum(np.square(theta[1:, :])) # Regularize

        # Gradient is the derivative of the cost function
        grad = (1.0 / m) * (X.T * (X*theta - y))
        grad[1:] = grad[1:] + (Lambda / m) * theta[1:] # Regularize
        
        return cost, np.ravel(grad)
        
    def __checkGrad(self, costFunction, theta):
        return None
        

class Poly2Regression(LinearRegression):

    """Regression model based on 2nd-degree polynomial regression"""
                
    def train(self, X, y):
        """Train the linear model using the polynomial features."""
        X = self.__addPolyFeatures(X)        
        super(Poly2Regression, self).train(X, y)

    def predict(self, X):
        """Make a prediction based on the most recent training session."""
        X = self.__addPolyFeatures(X)
        return super(Poly2Regression, self).predict(X)      
      
    def __addPolyFeatures(self, X):
        """Add the second-order polynomial combination of features."""
        
        # TODO: Make everything numpy!
        X = np.matrix(X)
        n = X.shape[1]
        for i in xrange(n):
            for j in xrange(i, n):
                feature = np.multiply(X[:, i], X[:, j])
                X = np.concatenate((X, feature), 1)

        return X
        
        