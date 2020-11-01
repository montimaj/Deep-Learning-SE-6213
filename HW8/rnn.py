import numpy as np
from layers import *
import layers
from rnn_layers import *
from utilities import *
from utilities import _get_batch,_divide_into_batches

class language_model(object):
    """docstring for language_model"""
    def __init__(self, data,**kwargs):
        
        
        self.train_data = data['train_data']
        self.valid_data = data['valid_data']
        self.test_data = data['test_data']
        self.vocabulary = data['vocabulary']
        self.word_to_idx = data['word_ids']
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        
        self.cell_type = kwargs.pop('cell_type','rnn')
        self.wordvec_dim = kwargs.pop('wordvec_dim',128)
        self.hidden_dim = kwargs.pop('hidden_dim',32)
        self.epochs = kwargs.pop('epochs',70)
        self.batch_size = kwargs.pop('batch_size',64)
        self.seq_len = kwargs.pop('seq_len',4)
        self.update_rule = kwargs.pop('update_rule','SGD')
        self.use_pre_trained = kwargs.pop('use_pre_trained',False)
        self.grads = {}

        
        if self.use_pre_trained:
            self.params = kwargs.pop('params',None)
            
        else:
            # Initialize word vectors
            self.params = {}
            self.params['W_embed'] = np.random.randn(self.vocabulary, self.wordvec_dim)
            self.params['W_embed'] /= 100

            # Initialize parameters for the RNN
            dim_mul = {'lstm': 4, 'rnn': 1}[self.cell_type]
            self.params['Wx'] = np.random.randn(self.wordvec_dim, dim_mul * self.hidden_dim)
            self.params['Wx'] /= np.sqrt(self.wordvec_dim)
            self.params['Wh'] = np.random.randn(self.hidden_dim, dim_mul * self.hidden_dim)
            self.params['Wh'] /= np.sqrt(self.hidden_dim)
            self.params['b'] = np.zeros(dim_mul * self.hidden_dim)

            # Initialize output to vocab weights
            self.params['W_vocab'] = np.random.randn(self.hidden_dim, self.vocabulary)
            self.params['W_vocab'] /= np.sqrt(self.hidden_dim)
            self.params['b_vocab'] = np.zeros(self.vocabulary)
        assert self.params, 'weights not initialized'

        self.hist = {}
        for key,_ in self.params.items():
            self.hist[key]  = {}


    def forward_pass(self,X,prev_h = None):
        cache = {}
        if prev_h is None:
            prev_h = np.zeros((self.batch_size,self.hidden_dim))
        x,cache['c_we'] = word_embedding_forward(X, self.params['W_embed'])
        if self.cell_type == 'rnn':
            x,cache['c_rnn'] = rnn_forward(x, prev_h, self.params['Wx'], self.params['Wh'], self.params['b'])
        else:
            x,cache['c_rnn'] = lstm_forward(x, prev_h, self.params['Wx'], self.params['Wh'], self.params['b'])
        x,cache['c_taf'] = temporal_affine_forward(x, self.params['W_vocab'], self.params['b_vocab'])
        return(x,cache)

    def backward_pass(self,scores,Y,cache):
        grads = {}
        loss, dscores = temporal_softmax_loss(scores, Y, mask = np.ones(Y.shape))
        dx, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dscores, cache['c_taf'])
        if self.cell_type == 'rnn':
            dcaptions_in_init, dhidden_init, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dx, cache['c_rnn'])
        else:
            dcaptions_in_init, dhidden_init, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dx, cache['c_rnn'])
        grads['W_embed'] = word_embedding_backward(dcaptions_in_init, cache['c_we'])
        return (loss,grads)

    def accuracy(self,X):
        batch_data = _divide_into_batches(X, self.batch_size)
        iterations = int(batch_data.shape[0])
        running_sum = 0
        for _iter in range(iterations):
            x, y = _get_batch(batch_data,_iter,self.seq_len)
            scores,_ = self.forward_pass(x)
            predictions = np.argmax(scores, axis = 2)
            running_sum += (predictions == y).sum()
        accuracy = running_sum/(self.batch_size*iterations)
        return(accuracy)


    def train(self):

        self.update_rule = getattr(layers, self.update_rule)
        batch_data = _divide_into_batches(self.train_data, self.batch_size)
        iterations = int(batch_data.shape[0])
        for epoch in range(self.epochs):
            print ('Epoch '+str(epoch+1) + '/'+ str(self.epochs))
            for _iter in range(iterations-1):
                X, Y = _get_batch(batch_data,_iter,self.seq_len)
                scores, cache = self.forward_pass(X)
                cost, grads = self.backward_pass(scores,Y,cache)
                for key,w in self.params.items():
                    self.params[key],self.hist[key] = self.update_rule(w,grads[key],self.hist[key])
                if _iter % 50 ==0:
                    print('    Iteration '+str(_iter+1) + '/'+ str(iterations) + ', Loss: ' + str(cost))
            tr_acc = self.accuracy(self.train_data)
            val_acc = self.accuracy(self.valid_data)
            print('    tr_acc: '+ str(tr_acc)+', val_acc: '+str(val_acc))
        return (self.params)

        
        
