import numpy as np
import math
class Logistic_regression:
    def __init__(self):
        pass
    def transform(self,X):
        self.p,self.q=X.shape
        X_trans=np.ones((self.p,1))
        X_trans=np.append(X_trans,X,axis=1)
        return X_trans
    def normalize(self,X):
        X[:,1:]=(X[:,1:]-np.mean(X[:,1:],axis=0))/np.std(X[:,1:],axis=0)
        return X
    def sigmoid(self,z):
        h=1/(1+math.e**(-z))
        return h
    def fit(self,X,y,iteration):
        self.X=X
        self.y=y
        self.iteration=iteration
        X_trans=self.transform(self.X)
        X_norm=self.normalize(self.X_trans)
        self.m,self.n=X_norm.shape
        self.theta=np.zeros((self.n,1))
        self.J=np.zeros((self.iteration,1))
        for i in range(self.iteration):
            h=self.sigmoid(np.dot(X_norm,self.theta))
            error=h-self.y
            self.theta=self.theta-(0.01/self.m)*np.dot(X_norm.T,error)
            self.J[i]=(-1/self.m)*((np.dot(y.T,math.log(h)))+np.dot((1-y).T,math.log(1-h)))
    def predict(self,X):
        x_transform=self.transform(X)
        x_normalize=self.normalize(x_transform)
        p=self.sigmoid(np.dot(x_normalize,self.theta))
        r,s=p.shape
        prediction=np.zeros((p.shape))
        for i in range(r):
            if(p[i]>=0.5):
                prediction[i]=1
            else:
                prediction[i]=0
        return prediction
            
            
            
    