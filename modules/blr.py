# Filename:     blr.py
# Authors:      apadin, dvorva, mjmor, mgkallit
# Start Date:   1/17/2017
# Last Update:  1/17/2017

"""
blr.py - Functions for training the Bayesian model

train - Accepts as input a training set X and labels y
        and returns the optimal parameters and hyper-parameters

"""


#==================== LIBRARIES ====================#
import numpy as np
import scipy as sp
import scipy.stats
from sklearn.linear_model import BayesianRidge

from stats import ewma


#==================== FUNCTIONS ====================#

# This function is used for training our Bayesian model
# Returns the regression parameters w_opt, and alpha, beta, S_N
# needed for the predictive distribution
# See p.152 of Bishop's "Pattern Recognition and Machine Learning"
def train(X, y):

    Phi = X # the measurement matrix of the input variables x (i.e., features)
    t   = y # the vector of observations for the target variable
    (N, M) = np.shape(Phi)
    # Init values for  hyper-parameters alpha, beta
    alpha = 5*10**(-3)
    beta = 5
    max_iter = 100
    k = 0

    PhiT_Phi = np.dot(np.transpose(Phi), Phi)
    s = np.linalg.svd(PhiT_Phi, compute_uv=0) # Just get the vector of singular values s

    ab_old = np.array([alpha, beta])
    ab_new = np.zeros((1,2))
    tolerance = 10**-3
    while( k < max_iter and np.linalg.norm(ab_old-ab_new) > tolerance):
        k += 1
        try:

            S_N = np.linalg.pinv(alpha*np.eye(M) + beta*PhiT_Phi)
        except np.linalg.LinAlgError as err:
            print  "******************************************************************************************************"
            print "                           ALERT: LinearAlgebra Error detected!"
            print "      CHECK if your measurement matrix is not leading to a singular alpha*np.eye(M) + beta*PhiT_Phi"
            print "                           GOODBYE and see you later. Exiting ..."
            print  "******************************************************************************************************"
            sys.exit(-1)

        m_N = beta * np.dot(S_N, np.dot(t, Phi).T).T
        gamma = sum(beta*s[i]**2 /(alpha + beta*s[i]**2) for i in range(M))
        #
        # update alpha, beta
        #
        ab_old = np.array([alpha, beta])
        alpha = float(gamma /np.inner(m_N,m_N))
        one_over_beta = 1/(N-gamma) * sum( (t[n] - np.inner(m_N, Phi[n]))**2 for n in range(N))
        beta = float(1/one_over_beta)
        ab_new = np.array([alpha, beta])

    S_N = np.linalg.pinv(alpha*np.eye(M) + float(beta)*PhiT_Phi)
    m_N = beta * np.dot(S_N, np.dot(t, Phi).T)
    w_opt = m_N

    return (w_opt, alpha, beta, S_N)
    
    
# See p.152 of Bishop's "Pattern Recognition and Machine Learning"
def sklearn_train(X, y):
    model = BayesianRidge().fit(X, y)
    beta = model.alpha_     # model.alpha_ is the noise precision ('beta' in Bishop)
    alpha = model.lambda_   # model.lambda_ is the weights precision ('alpha' in Bishop)
    
    PhiT_Phi = X.T * X
    M = X.shape[1]
    S_N = np.linalg.pinv(alpha*np.eye(M) + beta*PhiT_Phi)
    m_N = beta * np.dot(S_N, np.dot(y, X).T)
    w_opt = m_N

    return (w_opt, alpha, beta, S_N)
    

#==================== CLASSES ====================#
    
class Severity(object):
    
    def __init__(self, w=0.25, L=3, alert_count=2):
        self.beta = 1.0         # noise precision constant (determined by training)
        self.covariance = 0     # covariance (determined by training)
        self.avg_zscore = 0     # last average for EWMA chart
        self.alert_count = 0    # Number of alerts in a row
        self.set_wL(w, L)
        self.ALERT_THRESH = alert_count
    
    def update_params(self, beta, covariance):
        self.beta = beta
        self.covariance = covariance
        
    def set_wL(self, w, L):
        self.w = w  # EWMA weight
        self.L = L  # Std dev limit
        self.Z_THRESH = L * np.sqrt(w/(2-w))

    def check(self, error, x):
        mu = 0
        beta = self.beta
        S_N = self.covariance
        x = np.matrix(x).flatten()
        sigma = np.sqrt(1.0/beta + np.dot(x, np.dot(S_N, x.T)))
    
        if error < mu:  # Left-tailed
            p_value = sp.stats.norm.cdf(error, mu, sigma)
            zscore = sp.stats.norm.ppf(p_value) # inverse of cdf N(0,1)
        else:   # Right-tailed
            p_value = 1 - sp.stats.norm.cdf(error, mu, sigma)
            zscore = sp.stats.norm.ppf(1 - p_value) # inverse of cdf N(0,1)
            
        zscore = min(10, abs(zscore))   # Keep the zscore bounded
        self.avg_zscore = ewma(zscore, self.avg_zscore, self.w)  # EWMA
        
        # Detect anomalies with alert counter
        # A single alert is raised if the Z score is over the threshold
        if (self.avg_zscore > self.Z_THRESH):
            self.alert_count = self.alert_count + 1
        else:
            self.alert_count = 0

        # If several alerts are raised in succession, an anomly is reported
        if self.alert_count >= self.ALERT_THRESH:
            return True, zscore
        else:
            return False, zscore
