import numpy as np


def forward_step(x, w, b):
    """
    TO DO: Compute the forward pass of the neural network
    Points Allocated: 5

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

    out = x.dot(w) + b

    #### End of your code ####

    return (out, cache)


def backward_step(d_out, cache, input_layer=False):
    """
    TO DO: Compute the backward pass of the neural network
    Points Allocated: 15

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
    if not input_layer:
        # print('w', w.T)
        dx = d_out.dot(w.T)
    # print('x', x.T, 'd_out', d_out)
    dw = x.T.dot(d_out)
    db = np.sum(d_out, axis=0)

    #### End of your code ####

    return (dw, db, dx)


def ReLu_forward(x):
    """
    TO DO: Compute the ReLu activation for forward pass
    Points Allocated: 5

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
    Points Allocated: 15

    Inputs:
    d_out: derivatives of any shape
    cache: has x corresponding to d_out

    Outputs:
    dx: gradients with respect to x
    """
    x = cache
    dx = None
    ### Type your code here ###
    # print('relu_x', x)

    dx = d_out
    dx[x < 0] = 0

    #### End of your code ####
    return (dx)


def softmax(x):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    Points Allocated: 25

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
    ### Type your code here ###

    shift_scores = x - np.max(x, axis=1).reshape(-1, 1)
    out = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)

    #### End of your code ####
    return (out)


def loss(x, y):
    """
    TO DO: Compute the softmax loss and gradient for the neural network
    Points Allocated: 25

    Inputs:
    x: Matrix of shape (N,C) where N is number of samples
    and C is number of classes
    y: vector of labels with shape (N,)
    Outputs:
    loss: loss of forward pass (scalar value)
    dx: gradient of loss with respect to x
    """
    loss, de = None, None
    num_train = x.shape[0]
    loss = -np.sum(np.log(x[range(num_train), list(y)]))
    loss /= num_train
    de = x.copy()
    de[range(num_train), list(y)] += -1
    de /= num_train
    return (loss, de)

