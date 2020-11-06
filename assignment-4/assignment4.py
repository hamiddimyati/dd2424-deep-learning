__author__ = 'Hamid Dimyati'

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

np.random.seed(0)

def load_data(file_name):
    """
    Parameters:
        file_name: path where the file will be loaded
    
    Returns:
        compile_output: contains data, unique characters, mapping from character to index and reversely
    """
    with open(file_name, 'r') as file:
        data = file.read()
    chars = list(set(data))
    K = len(chars)
    char_to_ind = OrderedDict((char, idx) for idx, char in enumerate(chars))
    ind_to_char = OrderedDict((idx, char) for idx, char in enumerate(chars))
    compile_output = {'book_data':data, 'book_chars':chars, 'K':K, 'char_to_ind':char_to_ind, 'ind_to_char':ind_to_char}
    return compile_output

def onehot_conversion(compile_output, start, end):
    """
    Parameters:
        compile_output: contains data, unique characters, mapping from character to index and reversely
        start: index of the beginning of the chunk of the text
        end: index of the end of the chunk of the text

    Returns:
        X (K, n): a one-hot encoding matrix
    """
    X_chars = compile_output['book_data'][start:end]
    X_idx = [compile_output['char_to_ind'][k] for k in X_chars]
    X = np.zeros((compile_output['K'], len(X_chars)))
    for i in range(len(X_chars)):
        X[X_idx[i],i] = 1
    return X

def softmax(x):
    """
    Parameters:
        x: matrix to be transformed

    Returns:
        softmax transformation
    """
    a = np.exp(x - np.max(x, axis=0))
    return a / a.sum(axis=0)

class RecurrentNeuralNetwork():
    def __init__(self, m, K):
        """
        Parameters:
            m: the number of hidden nodes
            K: the number of unique labels
        """
        self.m = m
        self.K = K
        self.params = {}
        self.loss_train = []
        self.t = []

    def test_method(self, X, Y):
        """
        Parameters:
            X (K, n): the input matrix
            Y (K, n): the output matrix

        Returns:
            print the result of method validation between analytical and numerical approaches 
        """ 
        h=1e-4
        self.init_parameters(sig=0.01)
        h0 = np.zeros((self.m, 1))
        grads0, _, _ = self.compute_gradients(X, Y, h0)
        gu0, gw0, gv0, gb0, gc0 = grads0['U'], grads0['W'], grads0['V'], grads0['b'], grads0['c']
        epsilon = np.finfo(np.float64).eps

        # ComputeGradsNumSlow
        print('ComputeGradsNumSlow')
        grads1 = self.compute_grads_num_slow(X, Y, h=h)
        gu1, gw1, gv1, gb1, gc1 = grads1['U'], grads1['W'], grads1['V'], grads1['b'], grads1['c']

        gap_u = np.divide(np.abs(gu0-gu1), np.maximum(epsilon, (np.abs(gu0)) + (np.abs(gu1))))
        gap_w = np.divide(np.abs(gw0-gw1), np.maximum(epsilon, (np.abs(gw0)) + (np.abs(gw1))))
        gap_v = np.divide(np.abs(gv0-gv1), np.maximum(epsilon, (np.abs(gv0)) + (np.abs(gv1))))
        gap_b = np.divide(np.abs(gb0-gb1), np.maximum(epsilon, (np.abs(gb0)) + (np.abs(gb1))))
        gap_c = np.divide(np.abs(gc0-gc1), np.maximum(epsilon, (np.abs(gc0)) + (np.abs(gc1))))
        print("U: max {}, mean {}".format(np.max(gap_u), np.mean(gap_u)))
        print("W: max {}, mean {}".format(np.max(gap_w), np.mean(gap_w)))
        print("V: max {}, mean {}".format(np.max(gap_v), np.mean(gap_v)))
        print("b: max {}, mean {}".format(np.max(gap_b), np.mean(gap_b)))
        print("c: max {}, mean {}".format(np.max(gap_c), np.mean(gap_c)))

    def fit(self, compile_output, sig, seq_len, epoch, eta):
        """
        Parameters:
            compile_output: contains data, unique characters, mapping from character to index and reversely
            sig: the variance value for initializing the model parameters
            seq_len: the length of text to be trained
            epoch: number of iteration of training
            eta: the learning rate value
        """
        self.init_parameters(sig)
        self.AdaGrad(compile_output, seq_len, epoch, eta)

    def init_parameters(self, sig):
        """
        Parameters:
            sig: the variance value for initializing the model parameters
        """
        self.params['b'] = np.zeros((self.m, 1))
        self.params['c'] = np.zeros((self.K, 1))
        self.params['U'] = np.random.normal(0.0, sig, (self.m, self.K))
        self.params['W'] = np.random.normal(0.0, sig, (self.m, self.m))
        self.params['V'] = np.random.normal(0.0, sig, (self.K, self.m))

    def evaluate_rnn(self, h0, x):
        """
        Parameters:
            h0: initial hidden states
            x: input matrix
        
        Returns:
            a: linear combination of input
            h: next hidden states
            o: normalized hidden states
            p: probability of output
        """
        x = x.reshape(x.shape[0], 1) #(K,1)
        a = np.matmul(self.params['W'], h0) + np.matmul(self.params['U'], x) + self.params['b'] #(m,1)
        h = np.tanh(a) #(m,1)
        o = np.matmul(self.params['V'], h) + self.params['c'] #(K,1)
        p = softmax(o) #(K,1)
        return a, h, o, p

    def synthesize_text(self, h0, x0, n):
        """
        Parameters:
            h0: initial hidden states
            x0: input matrix
            n: number of character to be produced

        Returns:
            Y: one-hot encoding matrix of predicted text
        """
        x = x0
        h = h0
        Y = np.zeros((self.K, n))
        for t in range(n):
            _, h, _, p = self.evaluate_rnn(h, x)
            cp = np.cumsum(p)
            idx = np.random.choice(self.K, p=p.flat)
            x = np.zeros(x.shape)
            x[idx] = 1
            Y[idx,t] = 1
        return Y

    def compute_gradients(self, X, Y, h0):
        """
        Parameters:
            X (K, n): the input matrix
            Y (K, n): the output matrix
            h0: initial hidden states

        Returns:
            grads: gradient values of all hyper-parameters
            l: loss value
            hprev: previous hidden states
        """
        seq_len = X.shape[1]
        l = 0
        a, h, o, p = {}, {}, {}, {}
        h[-1] = h0
        grad_a, grad_h, grad_o = {}, {}, {}
        grad_U, grad_W, grad_V, grad_b, grad_c = np.zeros_like(self.params['U']), np.zeros_like(self.params['W']), np.zeros_like(self.params['V']), np.zeros_like(self.params['b']), np.zeros_like(self.params['c'])
        
        #forward-pass
        for t in range(seq_len):
            a[t], h[t], o[t], p[t] = self.evaluate_rnn(h[t-1], X[:,t]) #(m,1), (m,1), (K,1), (K,1)
            l += -np.log(np.matmul(Y[:,t].reshape(Y.shape[0], 1).T, p[t])) #(1,1)
        #backward-pass
        for t in reversed(range(seq_len)):
            grad_o[t] = -(Y[:,t].reshape(Y.shape[0], 1) - p[t]).T #(1,K)
            grad_V += np.matmul(grad_o[t].T, h[t].T) #(K,m)
            grad_c += grad_o[t].T #(K,1)
            if t == seq_len-1:
                grad_h[t] = np.matmul(grad_o[t], self.params['V']) #(1,m)
                grad_a[t] = np.multiply(grad_h[t], (1-np.tanh(a[t].T)**2)) #(1,m)
            else:
                grad_h[t] = np.matmul(grad_o[t], self.params['V']) + np.matmul(grad_a[t+1], self.params['W']) #(1,m)
                grad_a[t] = np.multiply(grad_h[t], (1-np.tanh(a[t].T)**2)) #(1,m)
            grad_W += np.matmul(grad_a[t].T, h[t-1].T) #(m,m)
            grad_U += np.matmul(grad_a[t].T, X[:,t].reshape(X.shape[0], 1).T) #(m,K)
            grad_b += grad_a[t].T #(m,1)
        
        grads = {
            'U': grad_U,
            'W': grad_W,
            'V': grad_V,
            'b': grad_b,
            'c': grad_c,
        }
        for g in grads:
            grads[g] = np.clip(grads[g], -5, 5)
        
        hprev = h[seq_len-1]

        return grads, l, hprev

    def compute_grads_num_slow(self, X, Y, h):
        """
        Parameters:
            X (K, n): the input matrix
            Y (K, n): the output matrix
            h: initial hidden states
        
        Returns:
            grads: gradient values of all hyper-parameters
        """
        grad_U, grad_W, grad_V, grad_b, grad_c = np.zeros_like(self.params['U']), np.zeros_like(self.params['W']), np.zeros_like(self.params['V']), np.zeros_like(self.params['b']), np.zeros_like(self.params['c'])
        h0 = np.zeros((self.m, 1))

        b_try = np.copy(self.params['b'])
        for i in range(len(self.params['b'])):
            self.params['b'] = np.array(b_try)
            self.params['b'][i] -= h
            _, c1, _ = self.compute_gradients(X, Y, h0)

            self.params['b'] = np.array(b_try)
            self.params['b'][i] += h
            _, c2, _ = self.compute_gradients(X, Y, h0)

            grad_b[i] = (c2-c1) / (2*h)
        self.params['b'] = b_try

        c_try = np.copy(self.params['c'])
        for i in range(len(self.params['c'])):
            self.params['c'] = np.array(c_try)
            self.params['c'][i] -= h
            _, c1, _ = self.compute_gradients(X, Y, h0)

            self.params['c'] = np.array(c_try)
            self.params['c'][i] += h
            _, c2, _ = self.compute_gradients(X, Y, h0)

            grad_c[i] = (c2-c1) / (2*h)
        self.params['c'] = c_try
            
        U_try = np.copy(self.params['U'])
        for i in np.ndindex(self.params['U'].shape):
            self.params['U']= np.array(U_try)
            self.params['U'][i] -= h
            _, c1, _ = self.compute_gradients(X, Y, h0)
                
            self.params['U'] = np.array(U_try)
            self.params['U'][i] += h
            _, c2, _ = self.compute_gradients(X, Y, h0)
                
            grad_U[i] = (c2-c1) / (2*h)
        self.params['U'] = U_try

        W_try = np.copy(self.params['W'])
        for i in np.ndindex(self.params['W'].shape):
            self.params['W']= np.array(W_try)
            self.params['W'][i] -= h
            _, c1, _ = self.compute_gradients(X, Y, h0)
                
            self.params['W'] = np.array(W_try)
            self.params['W'][i] += h
            _, c2, _ = self.compute_gradients(X, Y, h0)
                
            grad_W[i] = (c2-c1) / (2*h)
        self.params['W'] = W_try

        V_try = np.copy(self.params['V'])
        for i in np.ndindex(self.params['V'].shape):
            self.params['V']= np.array(V_try)
            self.params['V'][i] -= h
            _, c1, _ = self.compute_gradients(X, Y, h0)
                
            self.params['V'] = np.array(V_try)
            self.params['V'][i] += h
            _, c2, _ = self.compute_gradients(X, Y, h0)
                
            grad_V[i] = (c2-c1) / (2*h)
        self.params['V'] = V_try

        grads = {
            'U': grad_U,
            'W': grad_W,
            'V': grad_V,
            'b': grad_b,
            'c': grad_c,
        }
        for g in grads:
            grads[g] = np.clip(grads[g], -5, 5)

        return grads

    def AdaGrad(self, compile_output, seq_len, epoch, eta):
        """
        Parameters:
            compile_output: contains data, unique characters, mapping from character to index and reversely
            seq_len: the length of text to be trained
            epoch: number of iteration of training
            eta: the learning rate value

        Returns:
            print predicted text
        """
        n = len(compile_output['book_data'])
        nb_seq = int((n-1)/seq_len)
        t = 0
        smooth_loss = 0
        epsilon = np.finfo(np.float64).eps
        m_params = {
            'U': np.zeros_like(self.params['U']),
            'W': np.zeros_like(self.params['W']),
            'V': np.zeros_like(self.params['V']),
            'b': np.zeros_like(self.params['b']),
            'c': np.zeros_like(self.params['c'])
        }
        for i in range(epoch):
            e = 0
            hprev = np.zeros((self.m, 1))
            for j in range(nb_seq):
                if j == nb_seq-1:
                    X = onehot_conversion(compile_output, e, n-2)
                    Y = onehot_conversion(compile_output, e+1, n-1)
                    e = n
                else:
                    X = onehot_conversion(compile_output, e, e+seq_len)
                    Y = onehot_conversion(compile_output, e+1, e+seq_len+1)
                    e += seq_len

                grads, loss, h = self.compute_gradients(X, Y, hprev)
                if smooth_loss != 0:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                else:
                    smooth_loss = loss

                for g in grads:
                    m_params[g] += np.power(grads[g], 2)
                    self.params[g] -= np.multiply(eta / np.sqrt(m_params[g] + epsilon), grads[g])

                hprev = h

                if t % 100 == 0:
                    self.loss_train.append(smooth_loss[0][0])
                    self.t.append(t)

                if t % 1000 == 0:
                    print('t:', t, 'smooth_loss:', smooth_loss[0][0])

                if t % 10000 == 0:
                    Y_t = self.synthesize_text(hprev, X[:, 0], 200)
                    text = ''
                    for i in range(Y_t.shape[1]):
                        idx = np.where(Y_t[:,i]==1)[0][0]
                        text += compile_output['ind_to_char'][idx]
                    print(text)

                t += 1
        
        X = onehot_conversion(compile_output, 0, n-1)
        Y_t = self.synthesize_text(hprev, X[:, 0], 1000)
        text = ''
        for i in range(Y_t.shape[1]):
            idx = np.where(Y_t[:,i]==1)[0][0]
            text += compile_output['ind_to_char'][idx]
        print(text)

if __name__ == '__main__':
    start = time.time()
    data = load_data('goblet_book.txt')
    sig = 0.01
    seq_len = 25
    epoch = 7
    eta = 0.1
    X = onehot_conversion(data, 0, 25)
    Y = onehot_conversion(data, 1, 26)
    
    model_RNN = RecurrentNeuralNetwork(m=100, K=data['K'])
    model_RNN.test_method(X, Y)
    model_RNN.fit(data, sig, seq_len, epoch, eta)
    
    plt.plot(model_RNN.t, model_RNN.loss_train)
    plt.xlabel('Update step')
    plt.ylabel('Training loss')
    plt.title('Training loss by update step')
    plt.savefig('epoch vs loss for {} epochs.png'.format(epoch))
    plt.close()