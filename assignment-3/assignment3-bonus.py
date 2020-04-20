__author__ = "Hamid Dimyati"


import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

np.random.seed(6)

def load_meta(file):
    """
     Parameters:
        file: filename to be loaded

    Returns:
        dict: dictionary of meta/data   
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_batch(file):
    """
    Parameters:
        file: filename to be loaded

    Returns:
        X (d, n): image pixel data
        Y (K, n): one-hot representation of the label for each image
        y (n): label for each image
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data'].T
        y = dict[b'labels']
        Y = np.eye(10)[y].T
    return X, Y, y

def softmax(x):
    """
    Parameters:
        x: matrix to be transformed

    Returns:
        softmax transformation
    """
    a = np.exp(x - np.max(x, axis=0))
    return a / a.sum(axis=0)

def normalization(X, X_train, type_norm):
    """
    Parameters:
        X (d, n): image pixel data to be normalized
        X_train (d, n): training image pixel data as reference for normalization
        type_norm: the type of normalization, either z-score or min-max

    Returns:
        X (d, n): normalized image pixel data 
    """
    if type_norm == 'z-score':
        mean_X = np.mean(X_train, axis=1, keepdims=True)
        std_X = np.std(X_train, axis=1, keepdims=True)
        X = (X-mean_X)/std_X
    if type_norm == 'min-max':
        min_X = np.min(X_train, axis=1, keepdims=True)
        max_X = np.max(X_train, axis=1, keepdims=True)
        X = (X-min_X)/(max_X-min_X)
    return X


class NeuralNetwork():
    def __init__(self, n_hidden, batch_normalization, init_method, dropout, file='datasets/cifar-10-batches-py/batches.meta'):
        """
        Parameters:
            W1 (m, d): weight matrix from input layer to hidden layer
            W2 (K, m): weight matrix hidden layer to output layer
            b1 (m, 1): bias vector to hidden layer
            b2 (K, 1): bias vector to output layer
            cost_train: cost value during learning the training dataset
            loss_train: loss value during learning the training dataset
            acc_train: accuracy of the training dataset
            cost_val: cost value during learning the validation dataset
            loss_val: loss value during learning the validation dataset
            acc_val: accuracy of the validation dataset
            labels: label of each sample
        """
        self.W = {}
        self.b = {}
        self.gamma = {}
        self.beta = {}
        self.mean_av = {}
        self.variance_av = {}
        self.cost_train = []
        self.loss_train = []
        self.acc_train = []
        self.cost_val = []
        self.loss_val = []
        self.acc_val = []
        self.t = []
        self.labels = load_meta(file)[b'label_names']
        self.n_hidden = n_hidden
        self.batch_normalization = batch_normalization
        self.init_method = init_method
        self.dropout = dropout
        self.mask = {}
    
    def test_method(self, X, Y, lambda_, n_batch, type_norm):
        """
        Parameters:
            X (d, n): image pixel data 
            Y (K, n): one-hot representation of the label for each image 
            lambda_: value of regularization parameter
            n_batch: total samples per one batch

        Returns:
            print the result of method validation between analytical and numerical approaches 
        """ 
        n_layer = len(self.n_hidden) + 1
        X = normalization(X, X, type_norm)
        X_batch = X[:10, :n_batch]
        Y_batch = Y[:10, :n_batch]
        h=1e-6
        self.init_parameters(X_batch, sig=1e-1)
        grads0 = self.compute_gradients(X_batch, Y_batch, lambda_=0)
        gw0, gb0 = grads0['W'], grads0['b']
        if self.batch_normalization:
            ggamma0, gbeta0 = grads0['gamma'], grads0['beta']
        e = np.finfo(np.float64).eps

        # ComputeGradsNum
        print('ComputeGradsNum')
        grads1 = self.compute_grads_num(X_batch, Y_batch, lambda_=0, h=h)
        gw1, gb1 = grads1['W'], grads1['b']
        if self.batch_normalization:
            ggamma1, gbeta1 = grads1['gamma'], grads1['beta']
        for i in range(1,n_layer+1):
            gap_w = np.divide(np.abs(gw0[str(i)]-gw1[str(i)]), np.maximum(e, (np.abs(gw0[str(i)])) + (np.abs(gw1[str(i)]))))
            gap_b = np.divide(np.abs(gb0[str(i)]-gb1[str(i)]), np.maximum(e, (np.abs(gb0[str(i)])) + (np.abs(gb1[str(i)]))))
            print("W{0}: max {1}, mean {2}".format(i, np.max(gap_w), np.mean(gap_w)))
            print("b{0}: max {1}, mean {2}".format(i, np.max(gap_b), np.mean(gap_b)))
            if (self.batch_normalization) & (i < n_layer):
                gap_gamma = np.divide(np.abs(ggamma0[str(i)]-ggamma1[str(i)]), np.maximum(e, (np.abs(ggamma0[str(i)])) + (np.abs(ggamma1[str(i)]))))
                gap_beta = np.divide(np.abs(gbeta0[str(i)]-gbeta1[str(i)]), np.maximum(e, (np.abs(gbeta0[str(i)])) + (np.abs(gbeta1[str(i)]))))
                print("gamma{0}: max {1}, mean {2}".format(i, np.max(gap_gamma), np.mean(gap_gamma)))
                print("beta{0}: max {1}, mean {2}".format(i, np.max(gap_beta), np.mean(gap_beta)))

        # ComputeGradsNumSlow
        print('ComputeGradsNumSlow')
        grads2 = self.compute_grads_num(X_batch, Y_batch, lambda_=0, h=h)
        gw2, gb2 = grads2['W'], grads2['b']
        if self.batch_normalization:
            ggamma2, gbeta2 = grads2['gamma'], grads2['beta']
        for i in range(1,n_layer+1):
            gap_w = np.divide(np.abs(gw0[str(i)]-gw2[str(i)]), np.maximum(e, (np.abs(gw0[str(i)])) + (np.abs(gw2[str(i)]))))
            gap_b = np.divide(np.abs(gb0[str(i)]-gb2[str(i)]), np.maximum(e, (np.abs(gb0[str(i)])) + (np.abs(gb2[str(i)]))))
            print("W{0}: max {1}, mean {2}".format(i, np.max(gap_w), np.mean(gap_w)))
            print("b{0}: max {1}, mean {2}".format(i, np.max(gap_b), np.mean(gap_b)))
            if (self.batch_normalization) & (i < n_layer):
                gap_gamma = np.divide(np.abs(ggamma0[str(i)]-ggamma2[str(i)]), np.maximum(e, (np.abs(ggamma0[str(i)])) + (np.abs(ggamma2[str(i)]))))
                gap_beta = np.divide(np.abs(gbeta0[str(i)]-gbeta2[str(i)]), np.maximum(e, (np.abs(gbeta0[str(i)])) + (np.abs(gbeta2[str(i)]))))
                print("gamma{0}: max {1}, mean {2}".format(i, np.max(gap_gamma), np.mean(gap_gamma)))
                print("beta{0}: max {1}, mean {2}".format(i, np.max(gap_beta), np.mean(gap_beta)))
        

    def fit(self, X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_, keep_prob, n_batch, n_epoch, n_step, eta_min, eta_max, sig, shuffle, cost, cyclical, type_norm):
        """
        Parameters:
            X_train (d, n_1): image pixel training data
            Y_train (K, n_1): one-hot representation of the label for each training image
            y_train (n_1): label for each training image
            X_val (d, n_2): image pixel validation data
            Y_val (K, n_2): one-hot representation of the label for each validation image
            y_val (n_2): label for each validation image
            lambda_: value of regularization parameter
            n_batch: total samples per one batch
            eta: learning rate value
            n_epoch: total epoch
            shuffle (boolean): whether to apply random shuffling to the data
        """ 
        X_train_norm = normalization(X_train, X_train, type_norm)
        X_val_norm = normalization(X_val, X_train, type_norm)
        self.init_parameters(X_train_norm, sig)
        self.mini_batch_GD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, lambda_, keep_prob, n_batch, n_epoch, n_step, eta_min, eta_max, shuffle, cost, cyclical)

    def predict(self, X, y):
        """
        Parameters:
            X (d, n): image pixel data
            y (n): label for each image
        
        Returns:
            final_acc: accuracy of the model
        """
        final_acc = self.compute_accuracy(X, y, keep_prob=1.0, train_mode=False)
        return final_acc

    def montage(self, filename):
        """ Display the image for each label in W 
        Parameters:
            filename: filename

        Returns:
            save the figure in png format
        """
        fig, ax = plt.subplots(2,5)
        for i in range(2):
            for j in range(5):
                im  = self.W[5*i+j,:].reshape(32,32,3, order='F')
                sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                sim = sim.transpose(1,0,2)
                ax[i][j].imshow(sim, interpolation='nearest')
                ax[i][j].set_title("y="+str(5*i+j))
                ax[i][j].axis('off')
        fig.savefig('figure_{}.png'.format(filename))
        plt.close(fig)

    def plotting(self, filename):
        """
        Parameters:
            n_epoch: range of total epoch
            filename: filename
        
        Returns:
            save the figure in png format
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
        ax1.plot(self.t, self.loss_train, label='training loss')
        ax1.plot(self.t, self.loss_val, label='validation loss')
        ax1.legend(loc='best', fontsize='small')
        ax1.set_title('cross-entropy loss value in each step size')
        ax1.set_xlabel('step size', fontsize='small')
        ax1.set_ylabel('cross-entropy loss', fontsize='small')
        ax2.plot(self.t, self.cost_train, label='training cost')
        ax2.plot(self.t, self.cost_val, label='validation cost')
        ax2.legend(loc='best', fontsize='small')
        ax2.set_title('cost value in each step size')
        ax2.set_xlabel('step size', fontsize='small')
        ax2.set_ylabel('cost value', fontsize='small')
        ax3.plot(self.t, self.acc_train, label='training accuracy')
        ax3.plot(self.t, self.acc_val, label='validation accuracy')
        ax3.legend(loc='best', fontsize='small')
        ax3.set_title('accuracy in each step size')
        ax3.set_xlabel('step size', fontsize='small')
        ax3.set_ylabel('accuracy', fontsize='small')
        fig.savefig('plot_{}.png'.format(filename))
        plt.close(fig)

    def init_parameters(self, X, sig):
        """
        Parameters:
            X (d, n): image pixel data

        Returns:
            save the parameter initialization of W and b
        """
        d = X.shape[0]
        n_layer = len(self.n_hidden) + 1
        K = len(self.labels)
        dims = [d] + self.n_hidden + [K]
        for i in range(1,n_layer+1):
            if self.init_method=='He':
                self.W[str(i)] = np.random.normal(0.0, np.sqrt(2/dims[i-1]), (dims[i], dims[i-1]))
            elif self.init_method=='Xavier':
                self.W[str(i)] = np.random.normal(0.0, np.sqrt(1/dims[i-1]), (dims[i], dims[i-1]))
            else:
                self.W[str(i)] = np.random.normal(0.0, sig, (dims[i], dims[i-1]))
            
            self.b[str(i)] = np.zeros((dims[i], 1))
            if (self.batch_normalization) & (i < n_layer):
                self.gamma[str(i)] = np.ones(dims[i]).reshape(dims[i], 1)
                self.beta[str(i)] = np.zeros((dims[i], 1))
    
    def evaluate_classifier(self, X, keep_prob, train_mode):
        """
        Parameters:
            X (d, n): image pixel data
    
        Returns:
            P (K, n): probability of dot product of input data and parameters
        """
        n_layer = len(self.n_hidden) + 1
        s_layer = {}
        sh_layer = {}
        X_layer = {}
        alpha = 0.9

        X_layer['0'] = X
        for i in range(1,n_layer):
            if self.batch_normalization:
                s_layer[str(i)] = np.matmul(self.W[str(i)], X_layer[str(i-1)]) + self.b[str(i)]
                if train_mode:
                    mean_ = np.mean(s_layer[str(i)], axis=1, keepdims=True)
                    variance_ = np.var(s_layer[str(i)], axis=1, keepdims=True)
                    if self.mean_av.get(str(i)) is None:
                        self.mean_av[str(i)] = mean_ 
                    else:
                        self.mean_av[str(i)] = alpha * self.mean_av[str(i)] + (1-alpha) * mean_

                    if self.variance_av.get(str(i)) is None:
                        self.variance_av[str(i)] = variance_
                    else:
                        self.variance_av[str(i)] = alpha * self.variance_av[str(i)] + (1-alpha) * variance_ 
                sh_layer[str(i)] = (s_layer[str(i)] - self.mean_av[str(i)]) / np.sqrt(self.variance_av[str(i)] + np.finfo(np.float64).eps)
                si = np.multiply(self.gamma[str(i)], sh_layer[str(i)] + self.beta[str(i)]) 
            else:
                si = np.matmul(self.W[str(i)], X_layer[str(i-1)]) + self.b[str(i)]
            X_layer[str(i)] = np.maximum(0, si)
            if (self.dropout) & (train_mode):
                self.mask[str(i)] = (np.random.rand(X_layer[str(i)].shape[0], X_layer[str(i)].shape[1]) < keep_prob)
                X_layer[str(i)] = np.multiply(X_layer[str(i)], self.mask[str(i)]) 
                X_layer[str(i)] = X_layer[str(i)] / keep_prob
        
        s = np.matmul(self.W[str(n_layer)], X_layer[str(n_layer-1)]) + self.b[str(n_layer)]
        P = softmax(s)
        if self.batch_normalization:
            all_output = {
                'X_layer': X_layer,
                's_layer': s_layer,
                'sh_layer': sh_layer,
                'P': P
            }
        else:
            all_output = {
                'X_layer': X_layer,
                'P': P
            }
        return all_output

    def batch_norm_back_pass(self, l, G, s_layer):
        """
        Parameters:
            l: the index of the selected layer
            G (K, n): the gradient from the label data
            s_layer: probability value for the selected layer
            
        Returns:
            G (K, n): the gradient
        """
        n = G.shape[1]
        I = np.ones(n).reshape(n, 1)
        sigma1 = np.power(self.variance_av[str(l)] + np.finfo(np.float64).eps, -0.5)
        sigma2 = np.power(self.variance_av[str(l)] + np.finfo(np.float64).eps, -1.5)
        G1 = np.multiply(G, sigma1)
        G2 = np.multiply(G, sigma2)
        D = s_layer - self.mean_av[str(l)]
        c = np.matmul(np.multiply(G2, D), I)
        G = G1 - 1/n * np.matmul(G1, I) - 1/n * np.multiply(D, c)
        return G

    def compute_gradients(self, X, Y, lambda_, keep_prob):
        """
        Parameters:
            X (d, n): image pixel data
            Y (K, n): one-hot representation of the label for each image
            lambda_: value of regularization parameter
        
        Returns:
            grad_W (K, d): gradient of weight matrix
            grad_b (K, 1): gradient of bias vector
        """
        n = X.shape[1]
        I = np.ones(n).reshape(n, 1)
        n_layer = len(self.n_hidden) + 1
        grad_W = {}
        grad_b = {}
        if self.batch_normalization:
            grad_gamma = {}
            grad_beta = {}

            all_output = self.evaluate_classifier(X, keep_prob, train_mode=True)
            X_layer, s_layer, sh_layer, P = all_output['X_layer'], all_output['s_layer'], all_output['sh_layer'], all_output['P']
            G = -(Y-P)
            grad_W[str(n_layer)] = 1/n * np.matmul(G, X_layer[str(n_layer-1)].T) + 2 * lambda_ * self.W[str(n_layer)]
            grad_b[str(n_layer)] = 1/n * np.matmul(G, I)
            G = np.matmul(self.W[str(n_layer)].T, G)
            if self.dropout:
                G = np.multiply(G, self.mask[str(n_layer-1)])
                G = G / keep_prob
            G = np.multiply(G, X_layer[str(n_layer-1)] > 0.0)
            for i in range(n_layer-1, 0, -1):
                grad_gamma[str(i)] = 1/n * np.matmul(np.multiply(G, sh_layer[str(i)]), I)
                grad_beta[str(i)] = 1/n * np.matmul(G, I)
                G = np.multiply(G, np.matmul(self.gamma[str(i)], I.T))
                G = self.batch_norm_back_pass(i, G, s_layer[str(i)])
                grad_W[str(i)] = 1/n * np.matmul(G, X_layer[str(i-1)].T) + 2 * lambda_ * self.W[str(i)]
                grad_b[str(i)] = 1/n * np.matmul(G, I)
                G = np.matmul(self.W[str(i)].T, G)
                if (self.dropout) & (i > 1):
                    G = np.multiply(G, self.mask[str(i-1)])
                    G = G / keep_prob
                G = np.multiply(G, X_layer[str(i-1)] > 0.0)
            grads = {
                'W': grad_W,
                'b': grad_b,
                'gamma': grad_gamma,
                'beta': grad_beta
            }
        else:
            all_output = self.evaluate_classifier(X, keep_prob, train_mode=True)
            X_layer, P = all_output['X_layer'], all_output['P']
            G = -(Y-P)
            for i in range(n_layer, 1, -1):
                grad_W[str(i)] = 1/n * np.matmul(G, X_layer[str(i-1)].T) + 2 * lambda_ * self.W[str(i)]
                grad_b[str(i)] = 1/n * np.matmul(G, I)
                G = np.matmul(self.W[str(i)].T, G)
                if (self.dropout) & (i > 1):
                    G = np.multiply(G, self.mask[str(i-1)])
                    G = G / keep_prob
                G = np.multiply(G, X_layer[str(i-1)] > 0.0)
            grad_W['1'] = 1/n * np.matmul(G, X.T) + 2 * lambda_ * self.W['1']
            grad_b['1'] = 1/n * np.matmul(G, I)
            grads = {
                'W': grad_W,
                'b': grad_b
            }
        return grads

    def compute_grads_num(self, X, Y, lambda_, h):
        """ Converted from matlab code 
        Parameters:
            X (d, n): image pixel data
            Y (K, n): one-hot representation of the label for each image
            P (K, n): probability of dot product of input data and parameters 
            lambda_: value of regularization parameter
            h: small value as tolerable threshold
        
        Returns:
            grad_W (K, d): gradient of weight matrix
            grad_b (K, 1): gradient of bias vector
        """
        n_layer = len(self.n_hidden) + 1
        grad_W = {}
        grad_b = {}
        for i in range(1,n_layer+1):
            grad_W[str(i)] = np.zeros(self.W[str(i)].shape)
            grad_b[str(i)] = np.zeros(self.b[str(i)].shape)

        if self.batch_normalization:
            grad_gamma = {}
            grad_beta = {}
            for i in range(1,n_layer):
                grad_gamma[str(i)] = np.zeros(self.gamma[str(i)].shape)
                grad_beta[str(i)] = np.zeros(self.beta[str(i)].shape)
        
        c, _ = self.compute_cost(X, Y, lambda_);
        
        for j in range(1,n_layer+1):
            b_try = np.copy(self.b[str(j)])
            for i in range(len(self.b[str(j)])):
                self.b[str(j)] = np.array(b_try)
                self.b[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                grad_b[str(j)][i] = (c2-c) / h
            self.b[str(j)] = b_try
            
            W_try = np.copy(self.W[str(j)])
            for i in np.ndindex(self.W[str(j)].shape):
                self.W[str(j)] = np.array(W_try)
                self.W[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                grad_W[str(j)][i] = (c2-c) / h
            self.W[str(j)] = W_try
            
            if (self.batch_normalization) & (j < n_layer):
                gamma_try = np.copy(self.gamma[str(j)])
                for i in range(len(self.gamma[str(j)])):
                    self.gamma[str(j)] = np.array(gamma_try)
                    self.gamma[str(j)][i] += h
                    c2, _ = self.compute_cost(X, Y, lambda_)
                    grad_gamma[str(j)][i] = (c2-c) / h
                self.gamma[str(j)] = gamma_try

                beta_try = np.copy(self.beta[str(j)])
                for i in range(len(self.beta[str(j)])):
                    self.beta[str(j)] = np.array(beta_try)
                    self.beta[str(j)][i] += h
                    c2, _ = self.compute_cost(X, Y, lambda_)
                    grad_beta[str(j)][i] = (c2-c) / h
                self.beta[str(j)] = beta_try

        if self.batch_normalization:
            grads = {
                'W': grad_W,
                'b': grad_b,
                'gamma': grad_gamma,
                'beta': grad_beta
            }
        else:
            grads = {
                'W': grad_W,
                'b': grad_b
            }

        return grads

    def compute_grads_num_slow(self, X, Y, lambda_, h):
        """ Converted from matlab code 
        Parameters:
            X (d, n): image pixel data
            Y (K, n): one-hot representation of the label for each image
            P (K, n): probability of dot product of input data and parameters 
            lambda_: value of regularization parameter
            h: small value as tolerable threshold
        
        Returns:
            grad_W (K, d): gradient of weight matrix
            grad_b (K, 1): gradient of bias vector
        """
        n_layer = len(self.n_hidden) + 1
        grad_W = {}
        grad_b = {}
        for i in range(1,n_layer+1):
            grad_W[str(i)] = np.zeros(self.W[str(i)].shape)
            grad_b[str(i)] = np.zeros(self.b[str(i)].shape)
        
        if self.batch_normalization:
            grad_gamma = {}
            grad_beta = {}
            for i in range(1,n_layer):
                grad_gamma[str(i)] = np.zeros(self.gamma[str(i)].shape)
                grad_beta[str(i)] = np.zeros(self.beta[str(i)].shape)
        
        for j in range(1,n_layer+1):
            b_try = np.copy(self.b[str(j)])
            for i in range(len(self.b[str(j)])):
                self.b[str(j)] = np.array(b_try)
                self.b[str(j)][i] -= h
                c1, _ = self.compute_cost(X, Y, lambda_)

                self.b[str(j)] = np.array(b_try)
                self.b[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)

                grad_b[str(j)][i] = (c2-c1) / (2*h)
            self.b[str(j)] = b_try
            
            W_try = np.copy(self.W[str(j)])
            for i in np.ndindex(self.W[str(j)].shape):
                self.W[str(j)] = np.array(W_try)
                self.W[str(j)][i] -= h
                c1, _ = self.compute_cost(X, Y, lambda_)
                
                self.W[str(j)] = np.array(W_try)
                self.W[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                
                grad_W[str(j)][i] = (c2-c1) / (2*h)
            self.W[str(j)] = W_try

            if (self.batch_normalization) & (j < n_layer):
                gamma_try = np.copy(self.gamma[str(j)])
                for i in range(len(self.gamma[str(j)])):
                    self.gamma[str(j)] = np.array(gamma_try)
                    self.gamma[str(j)][i] -= h
                    c1, _ = self.compute_cost(X, Y, lambda_)

                    self.gamma[str(j)] = np.array(gamma_try)
                    self.gamma[str(j)][i] += h
                    c2, _ = self.compute_cost(X, Y, lambda_)

                    grad_gamma[str(j)][i] = (c2-c1) / (2*h)
                self.gamma[str(j)] = gamma_try

                beta_try = np.copy(self.beta[str(j)])
                for i in range(len(self.beta[str(j)])):
                    self.beta[str(j)] = np.array(beta_try)
                    self.beta[str(j)][i] -= h
                    c1, _ = self.compute_cost(X, Y, lambda_)

                    self.beta[str(j)] = np.array(beta_try)
                    self.beta[str(j)][i] += h
                    c2, _ = self.compute_cost(X, Y, lambda_)

                    grad_beta[str(j)][i] = (c2-c1) / (2*h)
                self.beta[str(j)] = beta_try

        if self.batch_normalization:
            grads = {
                'W': grad_W,
                'b': grad_b,
                'gamma': grad_gamma,
                'beta': grad_beta
            }
        else:
            grads = {
                'W': grad_W,
                'b': grad_b
            }
        
        return grads

    def compute_cost(self, X, Y, lambda_, keep_prob):
        """
        Parameters:
            X (d, n): image pixel data
            Y (K, n): one-hot representation of the label for each image
            lambda_: value of regularization parameter
        
        Returns:
            J: the sum of the loss of the network’s predictions with regularization term
            l: the sum of the loss function
        """
        n = X.shape[1]
        all_output = self.evaluate_classifier(X, keep_prob, train_mode=True)
        P = all_output['P']
        pre_l = np.matmul(Y.T, P)
        pre_l[pre_l == 0] = np.finfo(float).eps
        l = -1/n * np.log(pre_l).trace()
        W_total = 0
        for i in range(len(self.W)):
            W_total += np.sum(self.W[str(i+1)]**2)
        J = l + lambda_ * W_total
        return J, l
    
    def mini_batch_GD(self, X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_, keep_prob, n_batch, n_epoch, n_step, eta_min, eta_max, shuffle, cost, cyclical):
        """
        Parameters:
            X_train (d, n_1): image pixel training data
            Y_train (K, n_1): one-hot representation of the label for each training image
            y_train (n_1): label for each training image
            X_val (d, n_2): image pixel validation data
            Y_val (K, n_2): one-hot representation of the label for each validation image
            y_val (n_2): label for each validation image
            lambda_: value of regularization parameter
            n_batch: total samples per one batch
            eta: learning rate value
            n_epoch: total epoch
            shuffle (boolean): whether to apply random shuffling to the data
        """
        n = X_train.shape[1]
        n_layer = len(self.n_hidden) + 1
        eta = eta_min
        t = 0
        l = 0
        rec = 2 * n_step / 5
        for i in range(n_epoch):
            rand_id = np.random.permutation(n)
            for id in range(n // n_batch):
                if shuffle:
                    rand_batch_range = range(id * n_batch, ((id + 1) * n_batch))
                    batch_range = rand_id[rand_batch_range]
                else:
                    batch_range = range(id * n_batch, (id + 1) * n_batch)
                X_batch = X_train[:, batch_range]
                Y_batch = Y_train[:, batch_range]

                grads = self.compute_gradients(X_batch, Y_batch, lambda_, keep_prob)
                grad_W, grad_b = grads['W'], grads['b']
                if self.batch_normalization:
                    grad_gamma, grad_beta = grads['gamma'], grads['beta']
                for i in range(1,n_layer+1):
                    self.W[str(i)] -= eta * grad_W[str(i)]
                    self.b[str(i)] -= eta * grad_b[str(i)]
                    if (self.batch_normalization) & (i < n_layer):
                        self.gamma[str(i)] -= eta * grad_gamma[str(i)]
                        self.beta[str(i)] -= eta * grad_beta[str(i)]
                
                
                if cost & (t % rec == 0):
                    print(t)
                    cost_train, loss_train = self.compute_cost(X_train, Y_train, lambda_, keep_prob)
                    self.cost_train.append(cost_train)
                    self.loss_train.append(loss_train)
                    cost_val, loss_val = self.compute_cost(X_val, Y_val, lambda_, keep_prob)
                    self.cost_val.append(cost_val)
                    self.loss_val.append(loss_val)
                    acc_train = self.compute_accuracy(X_train, y_train, keep_prob, train_mode=True)
                    acc_val = self.compute_accuracy(X_val, y_val, keep_prob, train_mode=True)
                    self.acc_train.append(acc_train)
                    self.acc_val.append(acc_val)
                    self.t.append(t)
                
                t += 1

                if cyclical:
                    if (t + 1) % (2 * n_step) == 0:
                        l += 1
                
                    if (t >= 2 * l * n_step) & (t <= (2 * l + 1) * n_step):
                        eta = eta_min + ((t - (2 * l * n_step)) / n_step) * (eta_max - eta_min)
                    elif (t >= (2 * l + 1) * n_step) & (t <= 2 * (l + 1) * n_step):
                        eta = eta_max - ((t - ((2 * l + 1) * n_step)) / n_step) * (eta_max - eta_min)

    def compute_accuracy(self, X, y, keep_prob, train_mode):
        """
        Parameters:
            X (d, n): image pixel data
            y (n): label for each image

        Returns:
            acc: accuracy of the model
        """
        n = X.shape[1]
        all_output = self.evaluate_classifier(X, keep_prob, train_mode)
        P = all_output['P']
        k = np.argmax(P, axis=0).T
        count = k[k == np.asarray(y)].shape[0]
        acc = count/n
        return acc


if __name__ == "__main__":
    start = time.time()
    norm = 'z-score'

    X_train, Y_train, y_train = load_batch('datasets/cifar-10-batches-py/data_batch_1')
    # load full datasets
    
    for i in range(2,6):
        Xi, Yi, yi = load_batch('datasets/cifar-10-batches-py/data_batch_{}'.format(i))
        if i != 5:
            X_train = np.concatenate((X_train, Xi), axis=1)
            Y_train = np.concatenate((Y_train, Yi), axis=1)
            y_train = y_train + yi
        else:
            X_train = np.concatenate((X_train, Xi[:,:5000]), axis=1)
            Y_train = np.concatenate((Y_train, Yi[:,:5000]), axis=1)
            y_train = y_train + yi[:5000]
            X_val = Xi[:,5000:]
            Y_val = Yi[:,5000:]
            y_val = yi[5000:]
    
    #X_val, Y_val, y_val = load_batch('datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test, y_test = load_batch('datasets/cifar-10-batches-py/test_batch')
    X_train_norm = normalization(X_train, X_train, norm)
    X_val_norm = normalization(X_val, X_train, norm)
    X_test_norm = normalization(X_test, X_train, norm)

    n_batch=100
    eta_min=1e-5
    eta_max=1e-1

    n_epoch=20
    n_step=5 * 45000 / n_batch
    hid_nodes=[50, 50]
    BN = True
    
    lambda_ = 0.005
    
    model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He', dropout=False)
    model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, keep_prob=1.0 ,n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
    print(model_NN.predict(X_train_norm, y_train))
    print(model_NN.predict(X_val_norm, y_val))
    print(model_NN.predict(X_test_norm, y_test))
    
    # coarse search (2 cycles)
    lbd_list = np.arange(-100, -1, 5).tolist() + np.arange(-4, 0, 1).tolist()
    lambda_list = []
    acc_val_list = []
    for l in lbd_list:
        lambda_=10**l
        lambda_list.append(lambda_)
        model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He')
        model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
        acc = model_NN.predict(X_val_norm, y_val)
        acc_val_list.append(acc)
    print('all accuracy in coarse:')
    print(acc_val_list)
    top_l = np.asarray(lbd_list)[np.argsort(acc_val_list)[-3:][::-1]]
    print('top 3 lambda in coarse:')
    print(top_l)
    print(lbd_list)
    print(lambda_list)
    print(acc_val_list)
    plt.scatter(lbd_list, acc_val_list)
    plt.xlabel('Lambda 10^x')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy by Lambda [10^%.2f, 10^%.2f]' % (min(lbd_list), max(lbd_list)))
    plt.savefig('trial2 coarse lambda vs accuracy {}.png'.format(i))
    plt.close()
    # fine search (2 cycles)
    nb = 2
    l_min = -2.264 #-2.5
    l_max = -2.096 #-2.3
    lambda_dict = {}
    for i in range(3):
        lbd_list = []
        lambda_list = []
        acc_val_list = []
        for _ in range(15):
            lbd = l_min + (l_max - l_min) * np.random.rand(1,1)[0][0]
            lambda_=10**lbd
            lbd_list.append(lbd)
            lambda_list.append(lambda_)
            model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He')
            model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
            acc = model_NN.predict(X_val_norm, y_val)
            acc_val_list.append(acc)
        print(lbd_list)
        print(lambda_list)
        print(acc_val_list)
        plt.scatter(lbd_list, acc_val_list)
        plt.xlabel('Lambda 10^x')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy by Lambda [10^%.2f, 10^%.2f]' % (l_min, l_max))
        plt.savefig('trial3 lambda vs accuracy {}.png'.format(i))
        plt.close()
        lambda_dict[str(i)] = np.asarray(lambda_list)[np.argsort(acc_val_list)[-5:][::-1]]
        l_min = np.log10(np.min(lambda_dict[str(i)]))
        l_max = np.log10(np.max(lambda_dict[str(i)]))
        #winner = lambda_list[np.argmax(acc_val_list)]
        #lambda_list.sort()
        #l_min = np.log10(lambda_list[lambda_list.index(winner)-nb])
        #l_max = np.log10(lambda_list[lambda_list.index(winner)+nb])
    print('top 5 lambda in fine:')
    print(lambda_dict)
    with open('lambda_dict.p', 'wb') as fp:
        pickle.dump(lambda_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
      
    lambda_ = 0.00632131

    # varying number of hidden layers
    node_list = np.arange(1,11).tolist()
    acc_train_list = []
    acc_val_list = []
    acc_test_list = []
    for node in node_list:
        hid_nodes=[50] * node
        model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He', dropout=False)
        model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, keep_prob=1.0, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
        acc_train_list.append(model_NN.predict(X_train_norm, y_train))
        acc_val_list.append(model_NN.predict(X_val_norm, y_val))
        acc_test_list.append(model_NN.predict(X_test_norm, y_test))
    print('all accuracy:')
    print(acc_train_list)
    print(acc_val_list)
    print(acc_test_list)
    plt.plot(node_list, acc_train_list, label='training')
    plt.plot(node_list, acc_val_list, label='validation')
    plt.plot(node_list, acc_test_list, label='testing')
    plt.xlabel('number of hidden layers', fontsize='small')
    plt.ylabel('accuracy', fontsize='small')
    plt.title('accuracy by number of hidden layers')
    plt.legend(loc='best', fontsize='small')
    plt.savefig('accuracy vs no hidden layer.png')
    plt.close()
    
    # hidden layer size fine-tuning
    layer_list = []
    acc_val_list = []
    keep_p = 0.85
    for _ in range(20):
        hid_nodes=np.random.randint(50,200,size=4).tolist()
        layer_list.append(hid_nodes)
        model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He', dropout=False)
        model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, keep_prob=keep_p, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
        acc = model_NN.predict(X_val_norm, y_val)
        acc_val_list.append(acc)
    print(layer_list)
    print(acc_val_list)
    plt.scatter(range(len(layer_list)), acc_val_list)
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy by Hidden Layer Size')
    plt.savefig('hidden size vs accuracy.png')
    plt.close()
    
    # applying dropout
    hid_nodes = [195, 194, 129,  92]
    keep_p = 0.85
    model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He', dropout=True)
    model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, keep_prob=keep_p, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
    print(model_NN.predict(X_train_norm, y_train))
    print(model_NN.predict(X_val_norm, y_val))
    print(model_NN.predict(X_test_norm, y_test))
    
    # probability to keep the nodes fine-tuning
    hid_nodes = [195, 194, 129,  92]
    keep_p = np.arange(0.5, 1.0, 0.05).tolist()
    acc_train_list = []
    acc_val_list = []
    acc_test_list = []
    for p in keep_p:
        model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He', dropout=True)
        model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, keep_prob=p, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=False, cyclical=True, type_norm=norm)
        acc_train_list.append(model_NN.predict(X_train_norm, y_train))
        acc_val_list.append(model_NN.predict(X_val_norm, y_val))
        acc_test_list.append(model_NN.predict(X_test_norm, y_test))
    print(keep_p)
    print('all accuracy:')
    print(acc_train_list)
    print(acc_val_list)
    print(acc_test_list)
    plt.plot(keep_p, acc_train_list, label='training')
    plt.plot(keep_p, acc_val_list, label='validation')
    plt.plot(keep_p, acc_test_list, label='testing')
    plt.xlabel('keep probability', fontsize='small')
    plt.ylabel('accuracy', fontsize='small')
    plt.title('accuracy by keep probability')
    plt.legend(loc='best', fontsize='small')
    plt.savefig('accuracy vs keep probability.png')
    plt.close()
    
    # final model of 5-layer neural networks
    hid_nodes=[195, 194, 129, 92]
    BN = True
    keep_p = 0.75
    model_NN = NeuralNetwork(n_hidden=hid_nodes, batch_normalization=BN, init_method='He', dropout=True)
    model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, keep_prob=keep_p, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, sig=1e-1, shuffle=True, cost=True, cyclical=True, type_norm=norm)
    print('training accuracy: {}'.format(model_NN.predict(X_train_norm, y_train)))
    print('validation accuracy: {}'.format(model_NN.predict(X_val_norm, y_val)))
    print('testing accuracy: {}'.format(model_NN.predict(X_test_norm, y_test)))
    fname = "BN:{}, lambda:{}, n_batch:{}, n_epoch:{}, eta:{}, n_step:{} ".format(str(BN), lambda_, n_batch, n_epoch, 'cyclic', n_step) + str(hid_nodes)
    model_NN.plotting(filename=fname)