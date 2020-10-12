import numpy as np

def forward_step(x,w,b):
    """
    TO DO: Compute the forward pass of the neural network
    
    Inputs:
    x: Numpy array of shape (N,d) where N is number of samples 
    and d is the dimension of input
    w: numpy array of shape (d x H) where H is size of hidden layer
    b: It is the bias matrix of shape (H,)
    
    Outputs:
    out: output of shape (N x H)
    cache: values used to calculate the output (x,w,b)
    """
    out = None
    cache = (x,w,b)
    
    out = x.dot(w)+b
    
    return (out,cache)


def backward_step(d_out,cache,reg=0,input_layer = False):
    """
    TO DO: Compute the backward pass of the neural network
        
    Inputs:
    d_out: calculated derivatives 
    cache: (x,w,b) values of corresponding d_out values
    input_layer: TRUE/FALSE
    
    Outptus:
    dx: gradients with respect to x
    dw: gradients with respect to w
    db: gradients with respect to b 
    """
    x,w,b = cache
    dx,dw,db = None, None, None
    
    # Note that dx is not valid for input layer. If input_layer is true then dx will just return None
    
    
    if not input_layer:
        dx = d_out.dot(w.T)  
    
    db = np.sum(d_out, axis=0)
    dw = x.T.dot(d_out) + reg*w
    return (dw,db,dx)

def ReLu_forward(x):
    """
    TO DO: Compute the ReLu activation for forward pass
    
    Inputs:
    x: numpy array of any shape
    
    Outputs:
    out : should be same shape as input
    cache: values used to calculate the output (out)
    """
    cache = x
    out = None
    
    out = np.maximum(0, x)
    
    return (out,cache)

def ReLu_backward(d_out,cache):
    """
    TO DO: Compute the backward pass for ReLu 
    
    Inputs: 
    d_out: derivatives of any shape
    cache: has x corresponding to d_out
    
    Outputs:
    dx: gradients with respect to x
    """
    x = cache
    dx = None
    
    dx = d_out
    dx[x < 0] = 0
    
    return (dx)

def softmax(x):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    
    Inputs:
    x: numpy array of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    out: softmax output of shape 
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    out = None
    
    shift_scores = x - np.max(x, axis = 1).reshape(-1,1)
    out = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
       
    return (out)

def loss(x,y,reg=0,w_list=None):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    
    Inputs:
    x: Matrix of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    loss,de = None,None
    num_train = x.shape[0]
    loss = -np.sum(np.log(x[range(num_train), list(y)]))
    loss /= num_train 

    if reg > 0:
        w_sum = np.sum([np.sum(w_list[i]*w_list[i]) for i in range(len(w_list))])
        loss +=  reg * (1/len(w_list)) * w_sum

    de= x.copy()
    de[range(num_train), list(y)] += -1 
    de /= num_train
    return (loss,de)

def dropout_forward(x,probability,mode):
    '''
    TO DO: Compute the forward step for dropout
    Inputs: 
    x: numpy array of shape (N,D)
    probability: drop probability of a neuron
    mode: train/test as dropout happens only during train
    
    Outputs:
    out: numpy array of same shape as x
    cache: Values to be used during backward pass
    '''
    filter_ = None
    if mode=='train':
        [N,D] = x.shape
        filter_ = (np.random.rand(N,D) < (probability))/(probability) # with scaling
        out = x*filter_
    elif mode=='test':
        out = x
    else:
        raise ValueError("mode must be 'test' or 'train'")
    cache = (probability, filter_,mode)
    out = out.astype(x.dtype, copy=False)
    return(out,cache)

def dropout_backward(grad,cache):
    '''
    TO DO: Compute the backward step for dropout
    
    Inputs:
    grad: numpy array of shape (N,D)
    cache: Values which are calculated during dropout forward step
    
    Outputs:
    dx: numpy array of shape (N,D)
    '''
    _, filter_, mode = cache
    if mode == 'train':
        dx = grad*filter_
    elif mode == 'test':
        dx = grad
    else:
        raise ValueError("mode must be 'test' or 'train'")
    return(dx)

def SGD(w,dw,config):
    '''
    Function to update weights using SGD
    Inputs:
    w -  weights
    dw - gradients
    config - dictionary containing values
    Outputs:
    w - updated weights
    config - updated config file
    '''
    # initialize default values
    if not bool(config): 
        config.setdefault('learning_rate', 1e-2)
    w += -config['learning_rate']* dw
    return (w,config)

def SGD_with_momentum(w,dw,config):
    '''
    Function to update weights using SGD with momentum
    Inputs:
    w -  weights
    dw - gradients
    config - dictionary containing values
    Outputs:
    w - updated weights
    config - updated config file
    '''
    # initialize default values
    if not bool(config): 
        #print('Optimization using SGD and Momentum')
        config.setdefault('learning_rate', 1e-2)
        config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w)) # Initial Velocity
        
    ### Type your code here ###
    v = config['momentum'] * v - config['learning_rate'] * dw
    w += v
    #### End of your code ####
    
    config['velocity'] = v
    return (w,config)

def adagrad(w,dw,config):
    '''
    Function to update weights using SGD with momentum
    Inputs:
    w -  weights
    dw - gradients
    config - dictionary containing values
    Outputs:
    w - updated weights
    config - updated config file
    '''
    # initialize default values
    if not bool(config): 
        config.setdefault('learning_rate', 0.01)
        config.setdefault('delta', 1e-8)
        config.setdefault('r', np.zeros_like(w))
        
    ### Type your code here ###
    config['r'] += dw * dw
    w -= config['learning_rate'] * dw / (config['delta'] + np.sqrt(config['r']))
    #### End of your code ####
    
    return (w,config)
    
def rmsprop(w,dw,config):
    '''
    Function to update weights using SGD with momentum
    Inputs:
    w -  weights
    dw - gradients
    config - dictionary containing values
    Outputs:
    w - updated weights
    config - updated config file
    '''
    # initialize default values
    if not bool(config): 
        config.setdefault('learning_rate', 0.01)
        config.setdefault('decay_rate', 0.99)
        config.setdefault('delta', 1e-8)
        config.setdefault('r', np.zeros_like(w))
        
    ### Type your code here ###
    config['r'] = config['decay_rate'] * config['r'] + (1 - config['decay_rate']) * dw * dw
    w -= config['learning_rate'] * dw / np.sqrt(config['delta'] + config['r'])
    #### End of your code ####
    
    return (w,config)
    
def adam(w,dw,config):
    '''
    Function to update weights using SGD with momentum
    Inputs:
    w -  weights
    dw - gradients
    config - dictionary containing values
    Outputs:
    w - updated weights
    config - updated config file
    '''
    # initialize default values
    if not bool(config): 
        config.setdefault('learning_rate', 1e-2)
        config.setdefault('beta1', 0.9)
        config.setdefault('beta2', 0.999)
        config.setdefault('delta', 1e-8)
        config.setdefault('first_moment', np.zeros_like(w))
        config.setdefault('second_moment', np.zeros_like(w))
        config.setdefault('t', 0)
        
    ### Type your code here ###
    config['t'] += 1
    config['first_moment'] = config['beta1'] * config['first_moment'] + (1 - config['beta1']) * dw
    config['second_moment'] = config['beta2'] * config['second_moment'] + (1 - config['beta2']) * dw * dw
    bias_fm = config['first_moment'] / (1 - config['beta1'] ** config['t'])
    bias_sm = config['second_moment'] / (1 - config['beta2'] ** config['t'])
    w -= config['learning_rate'] * bias_fm / (config['delta'] + np.sqrt(bias_sm))
    #### End of your code ####
    
    return (w,config)

