from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Outputs:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    
    x_Wx = np.dot(x, Wx)
    prev_h_Wh = np.dot(prev_h, Wh)

    next_h = np.tanh(x_Wx + prev_h_Wh + b)

    cache = (Wx, Wh, b, x, prev_h, next_h)
    
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Output:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
   
    Wx, Wh, b, x, prev_h, next_h = cache

    dtanh = (1 - np.square(next_h)) * dnext_h

    db = np.sum(dtanh, axis=0)
    dWh = np.dot(prev_h.T, dtanh)
    dprev_h = np.dot(dtanh, Wh.T)
    dWx = np.dot(x.T, dtanh)
    dx = np.dot(dtanh, Wx.T)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Implement RNN forward on a sequence of data. 
    The input is a sequence composed of T vectors, each of dimension D.
    The RNN uses a hidden size of H, and we work over a minibatch containing N sequences

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Outputs:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    
    N, T, D = x.shape
    N, H = h0.shape

    h = np.zeros([N, T, H])
    cache = []

    prev_h = h0

    # RNN forward for T time steps.
    for t_step in range(T):
        cur_x = x[:, t_step, :]
        prev_h, cache_temp = rnn_step_forward(cur_x, prev_h, Wx, Wh, b)
        h[:, t_step, :] = prev_h
        cache.append(cache_temp)

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Outputs:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    Wx, Wh, b, x, prev_h, next_h = cache[0]
    N, T, H = dh.shape
    D, H = Wx.shape

    dx = np.zeros([N, T, D])
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(prev_h)

    for t_step in reversed(range(T)):

        cur_dh = dprev_h + dh[:,t_step,:]

        dx[:, t_step, :], dprev_h, dWx_temp, dWh_temp, db_temp = rnn_step_backward(cur_dh, cache[t_step])

        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp

    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Outputs:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    out = W[x, :]

    cache = x, W
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    Note: You can use the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Outputs:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None

    x, W = cache
    dW = np.zeros_like(W)

    np.add.at(dW, x, dout)

    return dW


def sigmoid(x):
    """
    logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    Forward pass for RNN/LSTM to make predictions

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range 0 <= y[i, t] < V

    Outputs:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and a minibatch size of N.
    Note: Use the sigmoid function provided above

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Outputs:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None

    ################# Type your code here #################
    H = prev_h.shape[1]
    U_z, U_r, U_c = Wx[:, :H], Wx[:, H: H * 2], Wx[:, H * 2: H * 3]
    W_z, W_r, W_c = Wh[:, :H], Wh[:, H: H * 2], Wh[:, H * 2: H * 3]
    b_z, b_r, b_c = b[:H], b[H: H * 2], b[H * 2: H * 3]
    z = sigmoid(np.dot(x, U_z) + np.dot(prev_h, W_z) + b_z)
    r = sigmoid(np.dot(x, U_r) + np.dot(prev_h, W_r) + b_r)
    next_c = np.tanh(np.dot(x, U_c) + np.dot(prev_h * r, W_c) + b_c)
    next_h = (1 - z) * next_c + z * prev_h
    cache = (x, Wx, Wh, b, prev_h, prev_c, z, r, next_c, next_h)

    ################# End of your code ####################

    # Cache variables needed for backprop.
    # cache = (x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h)

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Note: Implement sigmoid backward locally

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Outputs:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """

    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    ################# Type your code here #################

    x, Wx, Wh, b, prev_h, prev_c, z, r, next_c, next_h = cache
    H = dnext_h.shape[1]

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    U_z, U_r, U_c = Wx[:, :H], Wx[:, H: H * 2], Wx[:, H * 2: H * 3]
    W_z, W_r, W_c = Wh[:, :H], Wh[:, H: H * 2], Wh[:, H * 2: H * 3]

    dtanh = dnext_h * (1 - z) * (1 - np.square(next_c))
    db_c = np.sum(dtanh, axis=0)
    dU_c = np.dot(x.T, dtanh)
    dW_c = np.dot((next_h * r).T, dtanh)

    dsr = np.dot(dtanh, W_c.T)
    dsig_r = dsr * next_h * r * (1 - r)
    db_r = np.sum(dsig_r, axis=0)
    dU_r = np.dot(x.T, dsig_r)
    dW_r = np.dot(next_h.T, dsig_r)

    dz = dnext_h * (next_h - next_c)
    dsig_z = dz * z * (1 - z)
    db_z = np.sum(dsig_z, axis=0)
    dU_z = np.dot(x.T, dsig_z)
    dW_z = np.dot(next_h.T, dsig_z)

    dWx[:, :H], dWx[:, H: H * 2], dWx[:, H * 2: H * 3] = dU_z, dU_r, dU_c
    dWh[:, :H], dWh[:, H: H * 2], dWh[:, H * 2: H * 3] = dW_z, dW_r, dW_c
    db[:H], db[H: H * 2], db[H * 2: H * 3] = db_z, db_r, db_c

    dx = np.dot(dtanh, U_c.T) + np.dot(dsig_z, U_z.T) + np.dot(dsig_r, U_r.T)
    dprev_h = dnext_h * z + np.dot(dsig_z, W_z.T) + dsr * r + np.dot(dsig_r, W_r.T)

    ################# End of your code ####################
    

    return dx, dprev_h, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Implement LSTM forward on a sequence of data. 
    The input is a sequence composed of T vectors, each of dimension D.
    The LSTM uses a hidden size of H, and we work over a minibatch containing N sequences

    Note: The initial values for prev_h are set to 0 just like RNN

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    ################# Type your code here #################

    N, T, D = x.shape
    N, H = h0.shape

    h = np.zeros([N, T, H])
    cache = []

    prev_h = h0

    # RNN forward for T time steps.
    for t_step in range(T):
        cur_x = x[:, t_step, :]
        prev_h, prev_c, cache_temp = lstm_step_forward(cur_x, prev_h, 0, Wx, Wh, b)
        h[:, t_step, :] = prev_h
        cache.append(cache_temp)

    ################# End of your code ####################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Outputs:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ################# Type your code here #################

    x, Wx, Wh, b, prev_h, prev_c, z, r, next_c, next_h = cache[0]
    N, T, H = dh.shape
    D, H = Wx.shape

    dx = np.zeros([N, T, D])
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(prev_h)

    for t_step in reversed(range(T)):
        cur_dh = dprev_h + dh[:, t_step, :]

        dx[:, t_step, :], dprev_h, dWx_temp, dWh_temp, db_temp = lstm_step_backward(cur_dh, 0, cache[t_step])

        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp

    dh0 = dprev_h

    ################# End of your code ####################

    return dx, dh0, dWx, dWh, db


