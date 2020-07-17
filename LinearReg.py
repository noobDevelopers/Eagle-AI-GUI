import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class LinearReg:
    def __init__(self):
        self._mean = 0
        self._std  = 0
        self.theta = np.zeros((1,1))
        self.train_y=0
        self.train_x=0
        self.test_x = 0
        self.test_y = 0



    def _cal_cost(self, X, y, theta, lambd):
        m = y.shape[0]
        J = (1/(2*m))*(np.dot((np.dot(X,theta) - y).T,(np.dot(X,theta) - y)))
        regul = (lambd/(2*m))*np.sum(theta[1:]**2)
        J = J+regul
        return J[0][0]

    def _gradient(self, X, y, theta, lambd):
        m = y.shape[0]
        grad = (1/m)*(np.dot(X.T,(np.dot(X,theta) - y)))
        theta_regul = np.copy(theta)
        theta_regul[0] = 0
        regul = (lambd/m)*theta_regul
        grad = grad + regul
        return grad

    def train(self, data, data_test):
        X_vars = data.columns[0:len(data.columns)-1]
        y_vars = data.columns[len(data.columns)-1:]
        X = pd.DataFrame(data, columns = X_vars)
        y = pd.DataFrame(data, columns = y_vars)
        # print(X.head())
        self._mean  = X.mean()
        self._std = X.std()
        X = (X - self._mean)/self._std
        X['one'] = np.ones((X.shape[0],1))
        # print(X.head())
        a=X_vars.insert(0,'one') 
        X = X[a]
        # print(X.head())
        self.theta = np.zeros((X.shape[1],1))
        self.train_x = X
        self.train_y = y
        self.test_x = pd.DataFrame(data_test, columns = X_vars)
        self.test_y = pd.DataFrame(data_test, columns = y_vars)
        self.test_x = (self.test_x - self._mean)/self._std
        self.test_x['one'] = np.ones((self.test_x.shape[0],1))
        self.test_x = self.test_x[a]

    def cal_r2(self,y,y_pred):
        return r2_score(y, y_pred)

    def cal_rmse(self,y,y_pred):
        return mean_squared_error(y, y_pred, squared=False)

    def cal_mse(self, y, y_pred):
        return mean_squared_error(y, y_pred)


       
        

      


        
        
        
        
        

          
    
