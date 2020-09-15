import numpy as np
from utils import *
from layers import *

class n_layered_NN(object):
    '''
    The two_layered_nn contains all the logic required to implement a two layered neural network.
    '''
    def __init__(self,data,**kwargs):
        '''
        Create an instance to initialize all variables
        '''
        self.X_train = data['train']
        self.Y_train = data['tr_targets']
        self.X_val = data['val']
        self.Y_val = data['tr_val']
        self.n = kwargs.pop('n', 2)
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.batch_size = kwargs.pop('batch_size', 50)
        self.epochs = kwargs.pop('epochs',70)
        self.regularization_type = kwargs.pop('regularization_type','None')
        self.reg = kwargs.pop('reg',0)
        self.hidden_layer_sizes = kwargs.pop('hidden_layer_sizes',[100,100])
        assert len(self.hidden_layer_sizes) == self.n, "hidden layer sizes not provided for all layers"
        
        self.probability = kwargs.pop('keep_probability',0.8)
        self.classes = kwargs.pop('classes',10)

        self.parameters = {}
        for i in range(self.n):
            if i==0:
                self.parameters['w'+str(i)] = 0.3 * np.random.randn(np.int(self.X_train.shape[1]), self.hidden_layer_sizes[i])
            else:
                self.parameters['w'+str(i)] = 0.3 * np.random.randn(np.int(self.hidden_layer_sizes[i-1]), self.hidden_layer_sizes[i])
            self.parameters['b'+str(i)] = np.zeros(self.hidden_layer_sizes[i])
            if self.regularization_type == 'batchnorm':
                self.parameters['gamma' + str(i)] = np.ones(self.hidden_layer_sizes[i])
                self.parameters['beta' + str(i)] = np.zeros(self.hidden_layer_sizes[i])
        self.parameters['w'+str(self.n)] = 0.3 * np.random.randn(np.int(self.hidden_layer_sizes[self.n-1]), self.classes)
        self.parameters['b'+str(self.n)] = np.zeros(self.classes)
        self.bn_param = {}
        
    
    def forward_pass(self,X,mode):
        '''
        # Function to implement forward pass
        Input:
        - X:    input of shape NxD
        - mode: train/test
        Output:
        - scores: predictions of shape NxC
        - cache:  dictionary containing cache values 
        '''
        
        cache = {}
        self.bn_param['mode'] = mode
        
        for i in range(self.n):

            if i == 0:
                x = X
                
            x,cache['c'+str(i)] = forward_step(x,self.parameters['w'+str(i)],self.parameters['b'+str(i)])
            
            if self.regularization_type == 'batchnorm':
                x,cache['cb'+str(i)] = batchnorm_forward(x,self.parameters['gamma'+str(i)],self.parameters['beta'+str(i)],self.bn_param,i)
                          
            x,cache['cr'+str(i)] = ReLu_forward(x)
           
            if self.regularization_type == 'dropout':
                x,cache['cd'+str(i)] = dropout_forward(x,self.probability,mode)
            
        # add output layer    
        x,cache['c'+str(self.n)] = forward_step(x,self.parameters['w'+str(self.n)],self.parameters['b'+str(self.n)])   
        scores = softmax(x)
        
        return(scores,cache)
    
    def backward_pass(self,probs,y_batch,cache):
        '''
        #Function to implement backward pass
        
        Inputs:
        - probs:   softmax output of shape (NxC) 
        - y_batch: targets of shape (NxC)
        - cache:   cache values from forward pass
        Outputs:
        - loss_: loss value of the forward pass
        - grads: gradients of corresponding weights, biases, gamma and beta values
        '''
        grads = {}
        w_list = []
        # extract weights
        _ = [w_list.append(cache['c'+str(i)][1]) for i in range(self.n+1)]
        
        loss_,d = loss(probs,y_batch,self.reg,w_list) 
        
        grads['w'+str(self.n)],grads['b'+str(self.n)],d = backward_step(d,cache['c'+str(self.n)],self.reg)

        for i in reversed(range(self.n)):

            if self.regularization_type == 'dropout':
                d = dropout_backward(d,cache['cd'+str(i)])

            d = ReLu_backward(d,cache['cr'+str(i)])

            if self.regularization_type == 'batchnorm':
                d,grads['gamma'+str(i)],grads['beta'+str(i)] = batchnorm_backward(d,cache['cb'+str(i)])

            grads['w'+str(i)],grads['b'+str(i)],d = backward_step(d,cache['c'+str(i)],self.reg)

        
        return(loss_,grads)
    
    
    def predict(self,X):
        '''
        Function to predict the label for given input
        Inputs:
        - X: numpy array of shape NxD
        Outputs:
        - y:  numpy array of shape NxC
        '''
        probs,_ = self.forward_pass(X,'test')
        y = np.argmax(probs, axis = 1)
        return (y)
    
    def train(self):
        '''
        Function to train the two layered neural network
        Inputs:
        - none
        Outputs:
        - parameters:        variable containing trained weights, biases, gamma and beta values
        - loss_history:      list of loss values from training
        - train_acc_history: list of training accuracies
        - val_acc_history:   list of validation accuracies  
        '''
        print('***************************************************************')
        print('|                  Model Specifications:                      |')
        print('***************************************************************')
        print('input dimensions         :'+str(self.X_train.shape))
        print('output dimensions        :'+str(self.classes))
        print('number of hidden layers  :'+str(self.n))
        print('hidden layer dimensions  :'+str(self.hidden_layer_sizes))
        print('learning rate            :'+str(self.learning_rate))
        print('batch size               :'+str(self.batch_size))
        print('number of epochs         :'+str(self.epochs))
        print('---------------------------------------------------------------')
        if self.regularization_type == 'L2':
            print('regularization type      :L2')
            print('regularization constant  :'+str(self.reg))
        elif self.regularization_type == 'dropout':
            print('regularization type      :dropout')
            print('dropout keep probability :'+str(self.probability))
        elif self.regularization_type == 'batchnorm':
            print('regularization type      :batchnorm')
        else:
            print('regularization type      :None')
        print('---------------------------------------------------------------')
        
        
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]
        
        # Calculate iterations for each epoch
        _temp = np.int(self.batch_size)
        _iter = np.int(self.X_train.shape[0]/_temp)
        # train using the shuffled data
        for epoch in range(self.epochs):
            # shuffle the data
            idx = np.random.choice(np.int(self.X_train.shape[0]), np.int(self.X_train.shape[0]), replace=False)
            x_t = self.X_train[idx]
            y_t = self.Y_train[idx]
            # train for each batch
            for it in range(_iter):
                # extract a batch of data
                X_batch = x_t[_temp*it:_temp*it + _temp]
                y_batch = y_t[_temp*it:_temp*it + _temp]
                
                # The following steps implement the forward and backward pass
                probs,cache = self.forward_pass(X_batch,'train')
                loss_,grads = self.backward_pass(probs,y_batch,cache)
                loss_history.append(loss_)
                #Now update the weights, biases, gamma and beta values in paramaters
                for key, _ in self.parameters.items():
                    self.parameters[key] += -self.learning_rate*grads[key]
               
            # Calculaet training and validation accuracy 
            train_acc = (self.predict(self.X_train) == self.Y_train).mean()
            val_acc = (self.predict(self.X_val) == self.Y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            # print accuracies
            print ('Epoch '+str(epoch) + '/'+ str(self.epochs))
            print('Train accuracy: '+ str(train_acc) + ' Validation accuracy: '+ str(val_acc))
            
        return (self.parameters,loss_history,train_acc_history,val_acc_history)  
        
   