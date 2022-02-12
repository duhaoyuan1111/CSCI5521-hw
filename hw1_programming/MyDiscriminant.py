import numpy as np
import math

from numpy.core.fromnumeric import mean

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        xc1 = []
        xc2 = []
        for i in range(100):
            if ytrain[i] == 1:
                xc1.append(Xtrain[i])
            else:
                xc2.append(Xtrain[i])
        xc1 = np.array(xc1)
        xc2 = np.array(xc2)
        self.mean[0] = np.mean(xc1, axis=0)
        self.mean[1] = np.mean(xc2, axis=0)
        
        if self.shared_cov:
            self.S = np.cov(np.transpose(Xtrain),ddof=0)

        else:
            # compute the class-dependent covariance S1!=S2
            self.S[0] = np.cov(np.transpose(xc1),ddof=0)
            self.S[1] = np.cov(np.transpose(xc2),ddof=0)
            #print(self.S[0])
            #print(self.S[1])
            
    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        tempG_xC1 = 0
        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    # s1=s2
                    g_x = -1/2*np.matmul(np.matmul(np.transpose(Xtest[i] - self.mean[c]), np.linalg.inv(self.S)), Xtest[i] - self.mean[c]) + math.log(self.p[c])
                    if c == 1:
                        if g_x < tempG_xC1:
                            predicted_class[i] = 1
                        else :
                            predicted_class[i] = 2
                    tempG_xC1 = g_x
                else:
                    # s1!=s2
                    g_x = -1/2*math.log(np.linalg.det(self.S[c])) - 1/2*np.matmul(np.matmul(np.transpose(Xtest[i] - self.mean[c]), np.linalg.inv(self.S[c])), Xtest[i] - self.mean[c]) + math.log(self.p[c])
                    if c == 1:
                        if g_x < tempG_xC1:
                            predicted_class[i] = 1
                        else :
                            predicted_class[i] = 2
                    tempG_xC1 = g_x
        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance 1*8
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        xc1 = []
        xc2 = []
        for i in range(100):
            if ytrain[i] == 1:
                xc1.append(Xtrain[i])
            else:
                xc2.append(Xtrain[i])
        xc1 = np.array(xc1)
        xc2 = np.array(xc2)
        self.mean[0] = np.mean(xc1, axis=0)
        self.mean[1] = np.mean(xc2, axis=0)
        
        self.S = np.var(Xtrain,axis=0,ddof=0)
    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        g_x = 0
        g_x_part2 = 0
        tempG_xC1 = 0
        for t in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                g_x = pow(((Xtest[t]-self.mean[c])/np.sqrt(self.S)),2)
                g_x = g_x.sum()
                g_x_part2 = (-1/2) * g_x + math.log(self.p[c])
                if c == 1:
                    if tempG_xC1 > g_x_part2:
                        predicted_class[t] = 1
                    else:
                        predicted_class[t] = 2
                tempG_xC1 = g_x_part2
    
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
