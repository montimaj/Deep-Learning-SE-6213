from __future__ import print_function, division
import numpy as np
from builtins import range


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
    
    # out = x.dot(w)+b
    N = x.shape[0]
    x_temp = x.reshape(N,-1)
    out = x_temp.dot(w) + b
    
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
    
    '''
    if not input_layer:
        dx = d_out.dot(w.T)  
    
    db = np.sum(d_out, axis=0)
        
    ### Type your code here ###
    dw = x.T.dot(d_out) + reg*w
    
    #### End of your code ####
    '''
    db = np.sum(d_out, axis = 0)
    x_temp = x.reshape(x.shape[0],-1)
    dw = x_temp.T.dot(d_out)
    if not input_layer:
        dx = d_out.dot(w.T).reshape(x.shape)
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

    ### Type your code here ###
    if reg > 0:
        w_sum = np.sum([np.sum(w_list[i]*w_list[i]) for i in range(len(w_list))])
        loss +=  reg * (1/len(w_list)) * w_sum
    #### End of your code ####

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
        ### Type your code here ###
        [N,D] = x.shape
        filter_ = (np.random.rand(N,D) < (probability))/(probability) # with scaling
        out = x*filter_
        #filter_ = (np.random.rand(N,D) > probability
        #x[filter_] = 0
        #out = x
        #### End of your code ####
    elif mode=='test':
        ### Type your code here ###
        out = x
        #### End of your code ####
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
        ### Type your code here ###
        dx = grad*filter_
        #### End of your code ####
    elif mode == 'test':
        ### Type your code here ###
        dx = grad
        #### End of your code ####
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
    v = config['momentum']* v - config['learning_rate']*dw
    w +=v
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
    config['r'] = config['r']+(dw*dw)
    w += -config['learning_rate']* dw / (np.sqrt(config['r'])+config['delta'])
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
    config['r'] = config['decay_rate']*config['r']+(1-config['decay_rate'])*(dw*dw)
    w += -config['learning_rate']* dw / (np.sqrt(config['r'])+config['delta'])
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
    config['t']+=1 
    config['first_moment'] = config['beta1']*config['first_moment'] + (1- config['beta1'])*dw
    config['second_moment'] = config['beta2']*config['second_moment'] + (1- config['beta2'])*(dw**2)   
    first_moment_b = config['first_moment']/(1-config['beta1']**config['t'])
    second_moment_b = config['second_moment']/(1-config['beta2']**config['t'])
    w = w -config['learning_rate']* first_moment_b / (np.sqrt(second_moment_b) + config['delta'])
    #### End of your code ####
    return (w,config)


def conv_forward(x, w, b, stride, pad):
    """
    Implement forward pass for convolutional layer (20 pts)
    Input:
    - x: Input data of shape (N, D, H, W)
    - w: Filter weights of shape (F, D, h, w)
    - b: Biases, of shape (F,)
    - stride: The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - pad: The number of pixels that will be used to zero-pad the input.
    Outputs
    - out: Output of shape (N, F, H', W') where H' and W' are dependent on pad, stride and input values
    - cache: (x, w, b, stride, pad)
    """
    out = None
    
    N, D, H, W = x.shape
    F, _, h_, w_ = w.shape
    
    H_out = int(1 + (H + 2 * pad - h_) / stride)
    W_out = int(1 + (W + 2 * pad - w_) / stride)
    
    out = np.zeros((N,F,H_out,W_out)) 
    
    #### Type your code here ####
    
    x_pad = np.zeros((N,D,H+2*pad,W+2*pad))
    for n in range(N):
        for c in range(D):
            x_pad[n,c] = np.pad(x[n,c],(1,1),'constant', constant_values=(0,0))

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                for f in range(F):
                    current_x_matrix = x_pad[n,:, i*stride: i*stride+h_, j*stride:j*stride+w_]
                    current_filter = w[f] 
                    out[n,f,i,j] = np.sum(current_x_matrix*current_filter)
                out[n,:,i,j] = out[n,:,i,j]+b
    ### End of your code ###
    cache = (x, w, b, stride, pad)
    return out, cache


def conv_backward(dout, cache):
    """
    Implement backward pass for a convolutional layer. (20 pts)
    Inputs:
    - dout: Derivatives.
    - cache: A dictionary of (x, w, b, stride, pad) as in conv_forward
    Outputs:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #### Type your code here ####
    x, w, b, stride, pad = cache
    
    N, D, H, W = x.shape
    F, _, h_, w_ = w.shape
    _,_,H_out,W_out = dout.shape

    x_pad = np.zeros((N,D,H+2,W+2))
    for n in range(N):
        for c in range(D):
            x_pad[n,c] = np.pad(x[n,c],(1,1),'constant', constant_values=(0,0))

    db = np.zeros((F))
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                db = db + dout[n,:,i,j]

    dw = np.zeros(w.shape) 
    dx_pad = np.zeros(x_pad.shape)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    current_x_matrix = x_pad[n,:, i*stride: i*stride+h_, j*stride:j*stride+w_]
                    dw[f] = dw[f] + dout[n,f,i,j]* current_x_matrix
                    dx_pad[n,:, i*stride: i*stride+h_, j*stride:j*stride+w_] += w[f]*dout[n,f,i,j]

    dx = dx_pad[:,:,1:H+1,1:W+1]
    ### End of your code ###
    return dx, dw, db


def max_pool_forward(x, pool_height, pool_width, stride ):
    """
    Implement forward pass for a max pooling layer. (20 pts)
    Inputs:
    - x: Input data, of shape (N, D, H, W)
    - pool_height: The height of each pooling region
    - pool_width: The width of each pooling region
    - stride: The distance between adjacent pooling regions
    Output:
    - out: Output data
    - cache: (x, pool_height, pool_width, stride)
    """
    out = None
    #### Type your code here ####
    N, D, H, W = x.shape
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = np.zeros((N,D,H_out,W_out)) 

    for n in range(N):
        for c in range(D):
            for h in range(H_out):
                for w in range(W_out):
                    out[n,c,h,w] = np.max(x[n,c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width])
    ### End of your code ###
    cache = (x, pool_height, pool_width, stride)
    return out, cache


def max_pool_backward(dout, cache):
    
    """
    Implement backward pass for max pooling layer. (20 pts)
    Inputs:
    - dout: Derivatives
    - cache: Dictionary of (x, pool_height, pool_width, stride) as in max pool forward pass.
    Output:
    - dx: Gradient with respect to x
    """
    dx = None
    #### Type your code here ####
    x, pool_height, pool_width, stride = cache
    N,D,H_out,W_out = dout.shape

    dx = np.zeros(x.shape)

    for n in range(N):
        for c in range(D):
            for h in range(H_out):
                for w in range(W_out):
                    current_matrix= x[n,c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]
                    current_max = np.max(current_matrix)
                    for (i,j) in [(i,j) for i in range(pool_height) for j in range(pool_width)]:
                        if current_matrix[i,j] == current_max:
                            dx[n,c,h*stride+i,w*stride+j] += dout[n,c,h,w]


    ### End of your code ###
    return dx





