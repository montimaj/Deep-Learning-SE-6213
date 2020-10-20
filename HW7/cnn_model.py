import numpy as np
from layers import *
import layers
import datetime
import time

class n_layered_NN(object):
    '''
    The two_layered_nn contains all the logic required to implement a two layered neural network.
    '''
    def __init__(self,data,**kwargs):
        '''
        Create an instance to initialize all variables
        '''
        self.X_train = data['X_train'].astype('float64')
        self.Y_train = data['Y_train']
        self.X_val = data['X_val']
        self.Y_val = data['Y_val']

        self.n = kwargs.pop('n', 1)
        self.print_every = kwargs.pop('print_every', 1)

        self.num_filters = kwargs.pop('num_filters', 32)
        self.filter_size = kwargs.pop('filter_size', 7)
        self.hidden_dim = kwargs.pop('hidden_dim', 100)
        self.num_classes = kwargs.pop('num_classes', 10)
        self.weight_scale = kwargs.pop('weight_scale', 1e-3)

        self.update_rule = kwargs.pop('update_rule', 'SGD')
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 50)
        self.epochs = kwargs.pop('epochs',10)
        self.regularization_type = kwargs.pop('regularization_type','None')
        self.reg = kwargs.pop('reg',0)
        self.probability = kwargs.pop('keep_probability',0.8)

        self.parameters = {}
        self.hist = {}

        # convolutional layer parameters
        for i in range(self.n):
            self.parameters['w'+str(i)] = np.random.normal(0, self.weight_scale, [self.num_filters, 3, self.filter_size, self.filter_size]).astype('float64')
            self.hist['w'+str(i)] = {}
            self.parameters['b'+str(i)] = np.zeros(self.num_filters).astype('float64')
            self.hist['b'+str(i)] = {}
        
        # Dense layer parameters    
        self.parameters['w'+str(self.n)] = np.random.normal(0, self.weight_scale, [6272, self.hidden_dim]) #23328
        self.parameters['b'+str(self.n)] = np.zeros(self.hidden_dim)
        self.hist['w'+str(self.n)] = {}
        self.hist['b'+str(self.n)] = {}
        # Output layer parameters
        self.parameters['w'+str(self.n+1)] = np.random.normal(0, self.weight_scale, [self.hidden_dim, self.num_classes])
        self.parameters['b'+str(self.n+1)] = np.zeros(self.num_classes)
        self.hist['w'+str(self.n+1)] = {}
        self.hist['b'+str(self.n+1)] = {}

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
            
            x,cache['c'+str(i)] = conv_forward(x,self.parameters['w'+str(i)],self.parameters['b'+str(i)],1,1)
            
            x,cache['cr'+str(i)] = ReLu_forward(x)
            
            x,cache['po'+str(i)] = max_pool_forward(x,2,2,2)
           
            if self.regularization_type == 'dropout':
                x,cache['cd'+str(i)] = dropout_forward(x,self.probability,mode)
            
        # add dense layer 
        x,cache['c'+str(self.n)] = forward_step(x,self.parameters['w'+str(self.n)],self.parameters['b'+str(self.n)])   
        # add output layer   
        x,cache['c'+str(self.n+1)] = forward_step(x,self.parameters['w'+str(self.n+1)],self.parameters['b'+str(self.n+1)])  
        # add softmax layer
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
        
        grads['w'+str(self.n+1)], grads['b'+str(self.n+1)], d = backward_step(d,cache['c'+str(self.n+1)],self.reg)

        grads['w'+str(self.n)], grads['b'+str(self.n)], d = backward_step(d,cache['c'+str(self.n)],self.reg)

        for i in reversed(range(self.n)):

            if self.regularization_type == 'dropout':
                d = dropout_backward(d,cache['cd'+str(i)])

            d = max_pool_backward(d,cache['po'+str(i)])

            d = ReLu_backward(d,cache['cr'+str(i)])

            d, grads['w'+str(i)], grads['b'+str(i)] = conv_backward(d,cache['c'+str(i)])

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
        print('input dimensions                :'+str(self.X_train.shape))
        print('output dimensions               :'+str(self.num_classes))
        print('number of convolutional layers  :'+str(self.n))
        print('number of convolutional filters :'+str(self.num_filters))
        print('convolutional filter size       :'+str(self.filter_size))
        print('learning rate                   :'+str(self.learning_rate))
        print('batch size                      :'+str(self.batch_size))
        print('number of epochs                :'+str(self.epochs))
        print('---------------------------------------------------------------')
        if self.regularization_type == 'L2':
            print('regularization type             :L2')
            print('regularization constant         :'+str(self.reg))
        elif self.regularization_type == 'dropout':
            print('regularization type             :dropout')
            print('dropout keep probability :'+str(self.probability))
        elif self.regularization_type == 'batchnorm':
            print('regularization type             :batchnorm')
        else:
            print('regularization type             :None')
        print('---------------------------------------------------------------')
        print('optimization type               :' + self.update_rule)
        print('---------------------------------------------------------------')
        
        
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]
        self.update_rule = getattr(layers, self.update_rule)
        
        # Calculate iterations for each epoch
        _temp = np.int(self.batch_size)
        _iter = np.int(self.X_train.shape[0]/_temp)
        
        # train using the shuffled data
        for epoch in range(self.epochs):
            print ('Epoch '+str(epoch+1) + '/'+ str(self.epochs))
            # shuffle the data
            idx = np.random.choice(np.int(self.X_train.shape[0]), np.int(self.X_train.shape[0]), replace=False)
            x_t = self.X_train[idx]
            y_t = self.Y_train[idx]
            # train for each batch
            start_time = datetime.datetime.now()
            for it in range(_iter):
                # extract a batch of data
                X_batch = x_t[_temp*it:_temp*it + _temp]
                y_batch = y_t[_temp*it:_temp*it + _temp]
                
                # The following steps implement the forward and backward pass
                probs,cache = self.forward_pass(X_batch,'train')
                loss_,grads = self.backward_pass(probs,y_batch,cache)
                loss_history.append(loss_)
                
                #Now update the trainable values in paramaters
                for key,w in self.parameters.items():
                    self.parameters[key],self.hist[key] = self.update_rule(w,grads[key],self.hist[key])
                
                # print accuracies
                if _iter%self.print_every == 0:
                    print ('(Iteration '+str(it+1) + '/'+ str(_iter)+') loss: ' + str(loss_))
                
            
            # Calculaet training and validation accuracy 
            train_acc = (self.predict(self.X_train) == self.Y_train).mean()
            val_acc = (self.predict(self.X_val) == self.Y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            # print accuracies
            #print ('Epoch '+str(epoch) + '/'+ str(self.epochs))
            print('Train accuracy: '+ str(train_acc) + ' Validation accuracy: '+ str(val_acc))
            print('time for epoch: '+str(datetime.datetime.now() - start_time))
        return (self.parameters,loss_history,train_acc_history,val_acc_history)  
        
   