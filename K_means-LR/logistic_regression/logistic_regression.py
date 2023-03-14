import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    def __init__(self):
        #learning rate
        self.lr = 0.5
        #number of iterations 
        self.epochs = 100000
        self.w = None
        self.b = None
        
    def fit(self, X, y_true):
        m, n = X.shape
        # Model parameters
        self.w = np.zeros(n)
        self.b = 0
        for epoch in range(self.epochs):
            #linear model, dot product of weights and x, plus bias
            z = np.dot(X, self.w) + self.b
            y_pred = sigmoid(z)
            d = y_pred - y_true
            dw = (1 / m) * np.dot(X.T, d) 
            db = (1 / m) * np.sum(d) 

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
    
    def predict(self, z):
        return np.array(sigmoid(np.dot(z, self.w) + self.b))

def binary_accuracy(y_true, y_pred, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred)))

def sigmoid(z):
    return 1. / (1. + np.exp(-z))