__author__ = "Hamid Dimyati"

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(400)

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
        mean_X = np.mean(X_train, axis=1).reshape(-1,1)
        std_X = np.std(X_train, axis=1).reshape(-1,1)
        X = (X-mean_X)/std_X
    if type_norm == 'min-max':
        min_X = np.min(X_train, axis=1).reshape(-1,1)
        max_X = np.max(X_train, axis=1).reshape(-1,1)
        X = (X-min_X)/(max_X-min_X)
    return X


class NeuralNetwork():
    def __init__(self, loss_type, file='datasets/cifar-10-batches-py/batches.meta'):
        """
        Parameters:
            W (K, d): weight matrix to be optimized during learning process
            b (K, 1): bias vector to be optimized during learning process
            min_X: minimum value of training data for min-max normalization
            max_X: maximum value of training data for min-max normalization
            cost_train: cost value during learning the training dataset
            loss_train: loss value during learning the training dataset
            acc_train: accuracy of the training dataset
            cost_val: cost value during learning the validation dataset
            loss_val: loss value during learning the validation dataset
            acc_val: accuracy of the validation dataset
            labels: label of each sample
            loss_type: either 'cross-entropy' or 'svm-multiclass'
            margins: margin to calculate gradient of svm-multiclass loss function
        """
        self.W = None
        self.b = None
        self.cost_train = None
        self.loss_train = None
        self.acc_train = None
        self.cost_val = None
        self.loss_val = None
        self.acc_val = None
        self.labels = load_meta(file)[b'label_names']
        self.loss_type = loss_type
        self.margins = None
    
    def test_method(self, X, Y, lambda_, n_batch, ht, type_norm):
        """
        Parameters:
            X (d, n): image pixel data 
            Y (K, n): one-hot representation of the label for each image 
            lambda_: value of regularization parameter
            n_batch: total samples per one batch

        Returns:
            print the result of method validation between analytical and numerical approaches 
        """
        X = normalization(X, X, type_norm)
        X_batch = X[:20, :n_batch]
        Y_batch = Y[:20, :n_batch]
        h=1e-6
        self.init_parameters(X_batch)
        gw0, gb0 = self.compute_gradients(X_batch, Y_batch, lambda_=0)
        e = np.finfo(np.float32).eps
        # ComputeGradsNum
        gw1, gb1 = self.compute_grads_num(X_batch, Y_batch, lambda_=0, h=h)
        gap_w1 = np.divide(np.abs(gw0-gw1), np.maximum(e, (np.abs(gw0)) + (np.abs(gw1))))
        gap_b1 = np.divide(np.abs(gb0-gb1), np.maximum(e, (np.abs(gb0)) + (np.abs(gb1))))
        threshold_w1 = np.array([ht] * (gap_w1.shape[0] * gap_w1.shape[1])).reshape((gap_w1.shape[0], gap_w1.shape[1])) 
        threshold_b1 = np.array([ht] * (gap_b1.shape[0] * gap_b1.shape[1])).reshape((gap_b1.shape[0], gap_b1.shape[1])) 
        print('ComputeGradsNum')
        print("W's are equal: " + str(np.allclose(gap_w1, threshold_w1, rtol=0.0, atol=ht)))
        print("b's are equal: " + str(np.allclose(gap_b1, threshold_b1, rtol=0.0, atol=ht)))
        # ComputeGradsNumSlow
        gw2, gb2 = self.compute_grads_num_slow(X_batch, Y_batch, lambda_=0, h=h)
        gap_w2 = np.divide(np.abs(gw0-gw2), np.maximum(e, (np.abs(gw0)) + (np.abs(gw2))))
        gap_b2 = np.divide(np.abs(gb0-gb2), np.maximum(e, (np.abs(gb0)) + (np.abs(gb2))))
        threshold_w2 = np.array([ht] * (gap_w2.shape[0] * gap_w2.shape[1])).reshape((gap_w2.shape[0], gap_w2.shape[1])) 
        threshold_b2 = np.array([ht] * (gap_b2.shape[0] * gap_b2.shape[1])).reshape((gap_b2.shape[0], gap_b2.shape[1])) 
        print('ComputeGradsNumSlow')
        print("W's are equal: " + str(np.allclose(gap_w2, threshold_w2, rtol=0.0, atol=ht)))
        print("b's are equal: " + str(np.allclose(gap_b2, threshold_b2, rtol=0.0, atol=ht)))
        

    def fit(self, X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_, n_batch, eta, n_epoch, type_norm, shuffle, lr_decay):
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
        self.init_parameters(X_train_norm)
        self.mini_batch_GD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, lambda_, n_batch, eta, n_epoch, shuffle, lr_decay)

    def predict(self, X, y):
        """
        Parameters:
            X (d, n): image pixel data
            y (n): label for each image
        
        Returns:
            final_acc: accuracy of the model
        """
        final_acc = self.compute_accuracy(X, y)
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

    def plotting(self, n_epoch, filename):
        """
        Parameters:
            n_epoch: range of total epoch
            filename: filename
        
        Returns:
            save the figure in png format
        """
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
        ax1.plot(n_epoch, self.loss_train, label='training loss')
        ax1.plot(n_epoch, self.loss_val, label='validation loss')
        ax1.legend(loc='best', fontsize='small')
        ax1.set_title('cross-entropy loss value in each epoch')
        ax1.set_xlabel('epcoh', fontsize='small')
        ax1.set_ylabel('cross-entropy loss', fontsize='small')
        ax2.plot(n_epoch, self.cost_train, label='training cost')
        ax2.plot(n_epoch, self.cost_val, label='validation cost')
        ax2.legend(loc='best', fontsize='small')
        ax2.set_title('cost value in each epoch')
        ax2.set_xlabel('epcoh', fontsize='small')
        ax2.set_ylabel('cost value', fontsize='small')
        fig.savefig('plot_{}.png'.format(filename))
        plt.close(fig)

    def init_parameters(self, X):
        """
        Parameters:
            X (d, n): image pixel data

        Returns:
            save the parameter initialization of W and b
        """
        d = X.shape[0]
        K = len(self.labels)
        self.W = np.random.normal(0, 0.01, (K, d)).astype(np.float32)
        self.b = np.random.normal(0, 0.01, (K, 1)).astype(np.float32)

    def evaluate_classifier(self, X):
        """
        Parameters:
            X (d, n): image pixel data
    
        Returns:
            P (K, n): probability of dot product of input data and parameters
        """
        s = np.matmul(self.W, X) + self.b
        if self.loss_type=='cross-entropy':
            P = softmax(s)
        if self.loss_type=='svm-multiclass':
            P = s
        return P

    def compute_gradients(self, X, Y, lambda_):
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
        I = np.ones(n, dtype=np.float32).reshape(n, 1)

        P = self.evaluate_classifier(X)
        if self.loss_type=='cross-entropy':
            G = -(Y-P)
            grad_W = 1/n * np.matmul(G, X.T) + 2 * lambda_ * self.W
            grad_b = 1/n * np.matmul(G, I)
        if self.loss_type=='svm-multiclass':
            _, _ = self.compute_cost(X, Y, lambda_)
            binary = self.margins # (K,n)
            #binarize
            binary[self.margins > 0] = 1
            binary_col_sum = np.sum(binary, axis=0) # (,n)
            binary.T[np.arange(n), np.argmax(Y, axis=0)] = -binary_col_sum.T # (n,K) --> (,n)
            grad_W = 1/n * np.matmul(binary, X.T) + lambda_ * self.W
            grad_b = 1/n * np.matmul(binary, I)
        
        return grad_W, grad_b

    def compute_grads_num(self, X, Y, P, lambda_, h):
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
        grad_W = np.zeros(self.W.shape, dtype=np.float32);
        grad_b = np.zeros(self.b.shape, dtype=np.float32);

        c, _ = self.compute_cost(X, Y, lambda_);
        
        b_try = np.copy(self.b)
        for i in range(len(self.b)):
            self.b = np.array(b_try)
            self.b[i] += h
            c2, _ = self.compute_cost(X, Y, lambda_)
            grad_b[i] = (c2-c) / h

        self.b = b_try

        W_try = np.copy(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W = np.array(W_try)
                self.W[i,j] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                grad_W[i,j] = (c2-c) / h

        self.W = W_try

        return grad_W, grad_b

    def compute_grads_num_slow(self, X, Y, P, lambda_, h):
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
        grad_W = np.zeros(self.W.shape, dtype=np.float32);
        grad_b = np.zeros(self.b.shape, dtype=np.float32);
        
        b_try = np.copy(self.b)
        for i in range(len(self.b)):
            self.b = np.array(b_try)
            self.b[i] -= h
            c1, _ = self.compute_cost(X, Y, lambda_)
            
            self.b = np.array(b_try)
            self.b[i] += h
            c2, _ = self.compute_cost(X, Y, lambda_)
            
            grad_b[i] = (c2-c1) / (2*h)
        
        self.b = b_try
        
        W_try = np.copy(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W = np.array(W_try)
                self.W[i,j] -= h
                c1, _ = self.compute_cost(X, Y, lambda_)

                self.W = np.array(W_try)
                self.W[i,j] += h
                c2, _ = self.compute_cost(X, Y, lambda_)

                grad_W[i,j] = (c2-c1) / (2*h)
        
        self.W = W_try    
        
        return grad_W, grad_b

    def compute_cost(self, X, Y, lambda_):
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
        if self.loss_type=='cross-entropy':
            P = self.evaluate_classifier(X)
            pre_l = np.matmul(Y.T, P)
            pre_l[pre_l == 0] = np.finfo(np.float32).eps
            l = -1/n * np.log(pre_l).trace()

        if self.loss_type=='svm-multiclass':
            sj = self.evaluate_classifier(X) # (K,n)
            sy = sj.T[np.arange(n),np.argmax(Y, axis=0)].T # (,n)
            self.margins = np.maximum(0, sj-sy+1) # (K,n)
            self.margins.T[np.arange(n),np.argmax(Y, axis=0)] = 0
            l = 1/n * (np.sum(self.margins, axis=0)).sum()
        
        J = l + lambda_ * np.sum(self.W**2)
        return J, l

    def mini_batch_GD(self, X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_, n_batch, eta, n_epoch, shuffle, lr_decay):
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
        self.cost_train = np.zeros(n_epoch, dtype=np.float32)
        self.loss_train = np.zeros(n_epoch, dtype=np.float32)
        self.acc_train = np.zeros(n_epoch, dtype=np.float32)
        self.cost_val = np.zeros(n_epoch, dtype=np.float32)
        self.loss_val = np.zeros(n_epoch, dtype=np.float32)
        self.acc_val = np.zeros(n_epoch, dtype=np.float32)
        n = X_train.shape[1]
        for i in range(n_epoch):
            print(i)
            rand_id = np.random.permutation(n)
            for id in range(n // n_batch):
                if shuffle:
                    rand_batch_range = range(id * n_batch, ((id + 1) * n_batch))
                    batch_range = rand_id[rand_batch_range]
                else:
                    batch_range = range(id * n_batch, ((id + 1) * n_batch))
                X_batch = X_train[:, batch_range]
                Y_batch = Y_train[:, batch_range]

                grad_W, grad_b = self.compute_gradients(X_batch, Y_batch, lambda_)
                self.W -= eta * grad_W
                self.b -= eta * grad_b

            self.cost_train[i], self.loss_train[i] = self.compute_cost(X_train, Y_train, lambda_)
            self.cost_val[i], self.loss_val[i] = self.compute_cost(X_val, Y_val, lambda_)
            self.acc_train[i] = self.compute_accuracy(X_train, y_train)
            self.acc_val[i] = self.compute_accuracy(X_val, y_val)

            if lr_decay & (i+1) % 10 == 0:
                eta = eta * 0.9
                #print(eta)

    def compute_accuracy(self, X, y):
        """
        Parameters:
            X (d, n): image pixel data
            y (n): label for each image

        Returns:
            acc: accuracy of the model
        """
        n = X.shape[1]
        P = self.evaluate_classifier(X)
        k = np.argmax(P, axis=0).T
        count = k[k == np.asarray(y)].shape[0]
        acc = count/n
        return acc

if __name__ == "__main__":
    X_train, Y_train, y_train = load_batch('datasets/cifar-10-batches-py/data_batch_1')
    # load full datasets
    """
    for i in range(2,6):
        Xi, Yi, yi = load_batch('datasets/cifar-10-batches-py/data_batch_{}'.format(i))
        if i != 5:
            X_train = np.concatenate((X_train, Xi), axis=1)
            Y_train = np.concatenate((Y_train, Yi), axis=1)
            y_train = y_train + yi
        else:
            X_train = np.concatenate((X_train, Xi[:,:9000]), axis=1)
            Y_train = np.concatenate((Y_train, Yi[:,:9000]), axis=1)
            y_train = y_train + yi[:9000]
            X_val = Xi[:,9000:]
            Y_val = Yi[:,9000:]
            y_val = yi[9000:]
    """
    X_val, Y_val, y_val = load_batch('datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test, y_test = load_batch('datasets/cifar-10-batches-py/test_batch')

    # building model
    norm = 'z-score'
    loss = 'cross-entropy'
    #loss = 'svm-multiclass'
    model_NN = NeuralNetwork(loss_type=loss)
    X_train_norm = normalization(X_train, X_train, norm)
    X_val_norm = normalization(X_val, X_train, norm)
    X_test_norm = normalization(X_test, X_train, norm)
    
    list_lambda = [0, 0.1, 1]
    n_batch = [100]
    n_epoch = [40]
    list_eta = [0.1, 0.001]
    colnames = ["lambda", "n_batch", "n_epoch", "eta", "training_accuracy", "validation_accuracy", "testing_accuracy"]
    df = pd.DataFrame(columns=colnames)
    dict_result = {}
    for i in range(len(list_lambda)):
        for j in range(len(n_batch)):
            for k in range(len(n_epoch)):
                for l in range(len(list_eta)):
                    dict_result["lambda"] = list_lambda[i]
                    dict_result["n_batch"] = n_batch[j]
                    dict_result["n_epoch"] = n_epoch[k]
                    dict_result["eta"] = list_eta[l]
                    model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=list_lambda[i], n_batch=n_batch[j], n_epoch=n_epoch[k], eta=list_eta[l], type_norm=norm, shuffle=False, lr_decay=True)
                    dict_result["training_accuracy"] = model_NN.predict(X_train_norm, y_train)
                    dict_result["validation_accuracy"] = model_NN.predict(X_val_norm, y_val)
                    dict_result["testing_accuracy"] = model_NN.predict(X_test_norm, y_test)
                    df.loc[len(df)] = dict_result
                    
                    fname = "lambda:{}, n_batch:{}, n_epoch:{}, eta:{}".format(list_lambda[i], n_batch[j], n_epoch[k], list_eta[l])
                    model_NN.plotting(n_epoch=range(n_epoch[k]), filename=fname)
                    model_NN.montage(filename=fname)
    
    df.to_csv('result.csv', index=False)