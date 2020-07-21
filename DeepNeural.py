import numpy as np
import math
import pandas as pd
from DeepUtils import initialize_parameters,forward_propagation,compute_cost,backward_propagation,predict
class DeepNeuralNet:

    def __init__(self, data_train, data_test):

        self._mean = 0
        self._std = 0

        self.train_x,self.train_y = self.clean_data(data_train, train=True)
        self.test_x,self.test_y = self.clean_data(data_test)

        


    def clean_data(self,data,train=False):

        X_vars = data.columns[0:len(data.columns)-1]
        y_vars = data.columns[len(data.columns)-1:]
        X = pd.DataFrame(data, columns = X_vars)
        y = pd.DataFrame(data, columns = y_vars)
        if train == True:
            self._mean  = X.mean()
            self._std = X.std()
        X = (X - self._mean)/self._std
        X = X.T
        y = y.T
        X = X.values
        y = y.values

        return X,y

    def update_parameters_with_gd(self, parameters, grads, learning_rate):

        L = len(parameters) // 2

    
        for l in range(L):
        
            parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
            parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
        
            
        return parameters

    def random_mini_batches(self, X, Y, mini_batch_size = 64):
    
        m = X.shape[1]                 
        mini_batches = []
            
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        num_complete_minibatches = math.floor(m/mini_batch_size) 
        for k in range(0, num_complete_minibatches):

            mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
    
        if m % mini_batch_size != 0:
        
            mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:]
            mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:]
        
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def initialize_velocity(self, parameters):
    
        L = len(parameters) // 2 
        v = {}
        
        for l in range(L):
            
            v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        
            
        return v

    def update_parameters_with_momentum(self, parameters, grads, v, beta, learning_rate):

        L = len(parameters) // 2 
        
       
        for l in range(L):
            
           
            v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta)*grads["dW"+str(l+1)]
            v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta)*grads["db"+str(l+1)]
           
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
          
            
        return parameters, v

    def initialize_adam(self, parameters) :
        
        L = len(parameters) // 2 
        v = {}
        s = {}
        
    
        for l in range(L):

            v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
            s["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            s["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
    
        
        return v, s

    def update_parameters_with_adam(self, parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

        
        L = len(parameters) // 2                 
        v_corrected = {}                        
        s_corrected = {}                        
        
    
        for l in range(L):
            
            v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
            v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db"+str(l+1)]
        

        
            v_corrected["dW" + str(l+1)] = v["dW"+str(l+1)]/(1-(beta1)**t)
            v_corrected["db" + str(l+1)] = v["db"+str(l+1)]/(1-(beta1)**t)
        

            
            s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*(grads["dW"+str(l+1)]**2)
            s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*(grads["db"+str(l+1)]**2)
            
            s_corrected["dW" + str(l+1)] = s["dW"+str(l+1)]/(1-(beta2)**t)
            s_corrected["db" + str(l+1)] = s["db"+str(l+1)]/(1-(beta2)**t)
        

        
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon)
    

        return parameters, v, s

#     def model(self, X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
#           beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):


#         L = len(layers_dims)             
#         costs = []                    
#         t = 0                                               
#         m = X.shape[1]                  
        

#         parameters = initialize_parameters(layers_dims)


#         if optimizer == "gd":
#             pass 
#         elif optimizer == "momentum":
#             v = self.initialize_velocity(parameters)
#         elif optimizer == "adam":
#             v, s = self.initialize_adam(parameters)
        
        
#         for i in range(num_epochs):
            
        
#             minibatches = self.random_mini_batches(X, Y, mini_batch_size)
#             cost_total = 0
            
#             for minibatch in minibatches:

            
#                 (minibatch_X, minibatch_Y) = minibatch

            
#                 a3, caches = forward_propagation(minibatch_X, parameters)

            
#                 cost_total += compute_cost(a3, minibatch_Y)

            
#                 grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            
#                 if optimizer == "gd":
#                     parameters = self.update_parameters_with_gd(parameters, grads, learning_rate)
#                 elif optimizer == "momentum":
#                     parameters, v = self.update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
#                 elif optimizer == "adam":
#                     t = t + 1 
#                     parameters, v, s = self.update_parameters_with_adam(parameters, grads, v, s,
#                                                                 t, learning_rate, beta1, beta2,  epsilon)
#             cost_avg = cost_total / m
            
        
#             if print_cost and i % 1000 == 0:
#                 print ("Cost after epoch %i: %f" %(i, cost_avg))
#             if print_cost and i % 100 == 0:
#                 costs.append(cost_avg)
                    

#         return parameters

# data_train = pd.read_csv('log_train.csv')
# data_test = pd.read_csv('log_test.csv')
# deep_net  = DeepNeuralNet(data_train, data_test)
# # train 3-layer model
# layers_dims = [deep_net.train_x.shape[0], 5, 2, 1]
# parameters = deep_net.model(deep_net.train_x, deep_net.train_y, layers_dims, beta = 0.9, optimizer = "momentum")

# # Predict
# predictions = predict(deep_net.test_x, deep_net.test_y, parameters)