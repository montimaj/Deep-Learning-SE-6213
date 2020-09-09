import numpy as np
from sklearn.preprocessing import OneHotEncoder


def forward_step(x, w, b):
    """
    TO DO: Compute the forward step (wx+b) of the neural network
    
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
    cache = (x, w, b)
    ### Type your code here ###
    out = np.dot(w.T, x.T).T + b
    #### End of your code ####    
    return (out, cache)


def backward_step(d_out, cache, input_layer=False):
    """
    TO DO: Compute the corrsponding backward step of the neural network
    
    Inputs:
    d_out: calculated derivatives 
    cache: (x,w,b) values of corresponding d_out values
    input_layer: TRUE/FALSE
    
    Outptus:
    dx: gradients with respect to x
    dw: gradients with respect to w
    db: gradients with respect to b 
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    
    # Note that dx is not valid for input layer. If input_layer is true then dx will just return None
    
    ### Type your code here ###
    dw = np.dot(d_out.T, x).T
    db = np.sum(d_out.T, axis=1, keepdims=True).T
    if not input_layer:
        dx = np.dot(w, d_out.T).T
    #### End of your code ####
    
    return (dw, db, dx)


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
    ### Type your code here ###
    out = np.maximum(0, x)
    #### End of your code ####  
    return (out, cache)


def ReLu_backward(d_out, cache):
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
    ### Type your code here ###
    dx = np.full_like(d_out, fill_value=1)
    dx[x <= 0] = 0
    dx *= d_out
    #### End of your code #### 
    return (dx)


def softmax(x):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    
    Inputs:
    x: numpy array of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    out: softmax output 
    """
    out = None
    x = x - np.max(x, axis=1).reshape(-1, 1) # This is necessary to avoid large values during training.
    ### Type your code here ###    
    exp_x = np.exp(x)
    out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    #### End of your code ####     
    return (out)


def loss(x, y):
    """
    TO DO: Compute the loss and gradient for the output of neural network
    
    Inputs:
    x: Matrix of shape (N,C) where N is number of samples 
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    loss: loss of forward pass (scalar value)
    de: gradient of loss (de = dj/dx which was given in hw2_guide)
    """
    loss, de = None, None
    ### Type your code here ###
    n = y.shape[0]
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_transform = enc.fit_transform(y.reshape(n, -1))
    loss = -np.sum(y_transform * np.log(x)) / n
    de = x
    de[range(n), y] -= 1
    de /= x.shape[0]
    #### End of your code #### 
    return (loss, de)

