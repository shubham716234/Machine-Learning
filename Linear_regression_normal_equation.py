import numpy as np
import itertools
class Linear_regression:
    def __init__(self,degree):
        self.degree=degree
    def transform(self,X):
        self.p,self.q=X.shape
        X_transform=np.ones((self.p,1))
        X_transform=np.append(X_transform,X,axis=1)
        return X_transform
    def fit(self,X,y,iteration):
        self.X=X
        self.y=y
        self.iteration=iteration
        X_trans=self.transform(self.X)
        self.m,self.n=X_trans.shape
        self.theta=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X_trans),X_trans)),np.transpose(X_trans)),y)
        return self
    def predict(self,X):
        x_transform=self.transform(X)
        return np.dot(x_transform,self.theta)