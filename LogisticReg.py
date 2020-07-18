import pandas as pd
import numpy as np

class LogisticReg:
    def __init__(self, data_train, data_test):

         self._mean = 0
         self._std = 0

         self.train_x,self.train_y = self.clean_data(data_train, train=True)
         self.test_x,self.test_y = self.clean_data(data_test)

         self.theta = np.zeros((self.train_x.shape[1],1))


    def clean_data(self,data,train=False):

        X_vars = data.columns[0:len(data.columns)-1]
        y_vars = data.columns[len(data.columns)-1:]
        X = pd.DataFrame(data, columns = X_vars)
        y = pd.DataFrame(data, columns = y_vars)
        if train == True:
            self._mean  = X.mean()
            self._std = X.std()
        X = (X - self._mean)/self._std
        X['one'] = np.ones((X.shape[0],1))
        a=X_vars.insert(0,'one') 
        X = X[a]

        return X,y

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def cal_cost(self, X, y, theta, lambd):
       
        m = X.shape[0]
        h = self.sigmoid(np.dot(X,theta))
        J = (1/m)*(-1*np.dot(y.T,np.log(h)) - (np.dot((1-y).T,np.log(1-h))))
        reg = (lambd/(2*m))*(np.sum(theta[1:]**2))
        J = J+reg

        return np.squeeze(J)

    def cal_grad(self, X, y, theta, lambd):

        m = X.shape[0]
        h = self.sigmoid(np.dot(X,theta))
        grad = (1/m)*(np.dot(X.T,(h-y)))
        theta_reg = np.copy(theta)
        theta_reg[0] = 0
        reg = (lambd/m)*(theta_reg)
        grad = grad + reg

        return grad

    def predict(self, X, thresh):

        A = self.sigmoid(np.dot(X,self.theta))

        y_prediction = A
        y_prediction[y_prediction>thresh] = 1
        y_prediction[y_prediction<=thresh] = 0

        return y_prediction

    def prec_recall(self, y, y_pred):

        true_positives = np.sum((y==1)*(y_pred==1))
        predicted_positives = np.sum((y_pred == 1))
        prec = true_positives/predicted_positives

        actual_positives = np.sum((y==1))
        recall = true_positives/actual_positives

        return np.squeeze(prec),np.squeeze(recall)

    def cal_f1(self, prec, recall):
        return (2*prec*recall)/(prec+recall)

        

        


    
            
# data_train = pd.read_csv('log_train.csv')
# data_test = pd.read_csv('log_test.csv')
# logistic_reg = LogisticReg(data_train, data_test)


# w, X, Y = np.array([[1.],[2.],[2.]]), np.array([[1.,2.,-1.],[3.,4.,-3.2],[1,1,1]]), np.array([[1,0,1]])
# X =X.T
# Y = Y.T
# logistic_reg.cal_grad(X,Y,w,1)
