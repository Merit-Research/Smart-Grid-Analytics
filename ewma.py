# Classes for assesing severity and detecting anomalies
# Filename:     ewma.py
# Author:       apadin
# Start Date:   10/10/2016


#==================== CLASSES ====================#
class EWMA(object):

    def __init__(self, alpha, initial=0):
        self.last_sample = start
        self.alpha = alpha
        
    def check(self, sample):
        self.last_sample = (1.0 - self.alpha)*self.last_p + self.alpha*pvalue
        return self.last_sample