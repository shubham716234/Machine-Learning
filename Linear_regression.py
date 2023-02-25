import numpy as np;
import matplotlib.pyplot as plt
class Linear_regression:
    def __init__(self):
        pass
    def transform(self,X):
        self.p,self.q=X.shape
        X_transform=np.ones((self.p,1))
        X_transform=np.append(X_transform,X,axis=0);
        return X_transform
    def normalize(self,X):
        X[:,1:]=(X[:,1:]-np.mean(X[:,1:],axis=0))/np.std(X[:,1:],axis=0);
        return X
    def fit(self,X,y,iteration):
        self.X=X
        self.y=y
        self.iteration=iteration
        X_trans=self.transform(self.X)
        X_norm=self.normalize(X_trans)
        self.m,self.n=X_norm.shape
        J=np.zeros((self.iteration,1))
        self.theta=np.zeros((self.n, 1))
        for i in range(self.iteration):
            h=self.predict(X_norm)
            error=h-self.y
            J[i]=(1/2*self.m)*np.sum(np.square(error))
            self.theta=self.theta-(0.01/self.m)*np.dot(X_norm.T,error)
        return self
    def predict(self,X):
        return np.dot(X,self.theta)
    
    
