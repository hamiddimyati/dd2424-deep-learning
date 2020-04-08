__author__ = "Hamid Dimyati"


import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

np.random.seed(40)

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
    def __init__(self, n_hidden, file='datasets/cifar-10-batches-py/batches.meta'):
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
        self.cost_train = []
        self.loss_train = []
        self.acc_train = []
        self.cost_val = []
        self.loss_val = []
        self.acc_val = []
        self.t = []
        self.labels = load_meta(file)[b'label_names']
        self.n_hidden = n_hidden
    
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
        h=1e-5
        self.init_parameters(X_batch)
        gw10, gb10, gw20, gb20 = self.compute_gradients(X_batch, Y_batch, lambda_=0)
        e = np.finfo(np.float32).eps
        # ComputeGradsNum
        gw11, gb11, gw21, gb21 = self.compute_grads_num(X_batch, Y_batch, lambda_=0, h=h)
        gap_w11 = np.divide(np.abs(gw10-gw11), np.maximum(e, (np.abs(gw10)) + (np.abs(gw11))))
        gap_b11 = np.divide(np.abs(gb10-gb11), np.maximum(e, (np.abs(gb10)) + (np.abs(gb11))))
        threshold_w11 = np.array([ht] * (gap_w11.shape[0] * gap_w11.shape[1])).reshape((gap_w11.shape[0], gap_w11.shape[1])) 
        threshold_b11 = np.array([ht] * (gap_b11.shape[0] * gap_b11.shape[1])).reshape((gap_b11.shape[0], gap_b11.shape[1])) 
        gap_w21 = np.divide(np.abs(gw20-gw21), np.maximum(e, (np.abs(gw20)) + (np.abs(gw21))))
        gap_b21 = np.divide(np.abs(gb20-gb21), np.maximum(e, (np.abs(gb20)) + (np.abs(gb21))))
        threshold_w21 = np.array([ht] * (gap_w21.shape[0] * gap_w21.shape[1])).reshape((gap_w21.shape[0], gap_w21.shape[1])) 
        threshold_b21 = np.array([ht] * (gap_b21.shape[0] * gap_b21.shape[1])).reshape((gap_b21.shape[0], gap_b21.shape[1])) 
        print('ComputeGradsNum')
        print("W1's are equal: " + str(np.allclose(gap_w11, threshold_w11, rtol=0.0, atol=ht)))
        print("b1's are equal: " + str(np.allclose(gap_b11, threshold_b11, rtol=0.0, atol=ht)))
        print("W2's are equal: " + str(np.allclose(gap_w21, threshold_w21, rtol=0.0, atol=ht)))
        print("b2's are equal: " + str(np.allclose(gap_b21, threshold_b21, rtol=0.0, atol=ht)))
        # ComputeGradsNumSlow
        gw12, gb12, gw22, gb22 = self.compute_grads_num_slow(X_batch, Y_batch, lambda_=0, h=h)
        gap_w12 = np.divide(np.abs(gw10-gw12), np.maximum(e, (np.abs(gw10)) + (np.abs(gw12))))
        gap_b12 = np.divide(np.abs(gb10-gb12), np.maximum(e, (np.abs(gb10)) + (np.abs(gb12))))
        threshold_w12 = np.array([ht] * (gap_w12.shape[0] * gap_w12.shape[1])).reshape((gap_w12.shape[0], gap_w12.shape[1])) 
        threshold_b12 = np.array([ht] * (gap_b12.shape[0] * gap_b12.shape[1])).reshape((gap_b12.shape[0], gap_b12.shape[1])) 
        gap_w22 = np.divide(np.abs(gw20-gw22), np.maximum(e, (np.abs(gw20)) + (np.abs(gw22))))
        gap_b22 = np.divide(np.abs(gb20-gb22), np.maximum(e, (np.abs(gb20)) + (np.abs(gb22))))
        threshold_w22 = np.array([ht] * (gap_w22.shape[0] * gap_w22.shape[1])).reshape((gap_w22.shape[0], gap_w22.shape[1])) 
        threshold_b22 = np.array([ht] * (gap_b22.shape[0] * gap_b22.shape[1])).reshape((gap_b22.shape[0], gap_b22.shape[1])) 
        print('ComputeGradsNumSlow')
        print("W1's are equal: " + str(np.allclose(gap_w12, threshold_w12, rtol=0.0, atol=ht)))
        print("b1's are equal: " + str(np.allclose(gap_b12, threshold_b12, rtol=0.0, atol=ht)))
        print("W2's are equal: " + str(np.allclose(gap_w22, threshold_w22, rtol=0.0, atol=ht)))
        print("b2's are equal: " + str(np.allclose(gap_b22, threshold_b22, rtol=0.0, atol=ht)))
        

    def fit(self, X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_, n_batch, n_epoch, n_step, eta_min, eta_max, shuffle, cost, cyclical, type_norm):
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
        self.mini_batch_GD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, lambda_, n_batch, n_epoch, n_step, eta_min, eta_max, shuffle, cost, cyclical)

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

    def init_parameters(self, X):
        """
        Parameters:
            X (d, n): image pixel data

        Returns:
            save the parameter initialization of W and b
        """
        d = X.shape[0]
        m = self.n_hidden
        K = len(self.labels)
        self.W['1'] = np.random.normal(0, 1/np.sqrt(d, dtype=np.float32), (m, d)).astype(np.float32)
        self.W['2'] = np.random.normal(0, 1/np.sqrt(m, dtype=np.float32), (K, m)).astype(np.float32)
        self.b['1'] = np.zeros((m, 1), dtype=np.float32)
        self.b['2'] = np.zeros((K, 1), dtype=np.float32)

    def evaluate_classifier(self, X):
        """
        Parameters:
            X (d, n): image pixel data
    
        Returns:
            P (K, n): probability of dot product of input data and parameters
        """
        s1 = np.matmul(self.W['1'], X) + self.b['1']
        H = np.maximum(0, s1)
        s = np.matmul(self.W['2'], H) + self.b['2']
        P = softmax(s)
        return H, P

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
        
        H, P = self.evaluate_classifier(X)
        G = -(Y-P)
        grad_W2 = 1/n * np.matmul(G, H.T) + 2 * lambda_ * self.W['2']
        grad_b2 = 1/n * np.matmul(G, I)
        
        G = np.matmul(self.W['2'].T, G)
        #H[H <= 0.0] = 0.0
        G = np.multiply(G, H > 0.0)
        grad_W1 = 1/n * np.matmul(G, X.T) + 2 * lambda_ * self.W['1']
        grad_b1 = 1/n * np.matmul(G, I)
        return grad_W1, grad_b1, grad_W2, grad_b2

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
        grad_W = [np.zeros(self.W['1'].shape, dtype=np.float32), np.zeros(self.W['2'].shape, dtype=np.float32)];
        grad_b = [np.zeros(self.b['1'].shape, dtype=np.float32), np.zeros(self.b['2'].shape, dtype=np.float32)];

        c, _ = self.compute_cost(X, Y, lambda_);
        
        for j in range(1,3):
            b_try = np.copy(self.b[str(j)])
            for i in range(len(self.b[str(j)])):
                self.b[str(j)] = np.array(b_try)
                self.b[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                grad_b[j-1][i] = (c2-c) / h
            self.b[str(j)] = b_try
            
            W_try = np.copy(self.W[str(j)])
            for i in np.ndindex(self.W[str(j)].shape):
                self.W[str(j)] = np.array(W_try)
                self.W[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                grad_W[j-1][i] = (c2-c) / h
            self.W[str(j)] = W_try

        return grad_W[0], grad_b[0], grad_W[1], grad_b[1]

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
        grad_W = [np.zeros(self.W['1'].shape, dtype=np.float32), np.zeros(self.W['2'].shape, dtype=np.float32)];
        grad_b = [np.zeros(self.b['1'].shape, dtype=np.float32), np.zeros(self.b['2'].shape, dtype=np.float32)];
        
        for j in range(1,3):
            b_try = np.copy(self.b[str(j)])
            for i in range(len(self.b[str(j)])):
                self.b[str(j)] = np.array(b_try)
                self.b[str(j)][i] -= h
                c1, _ = self.compute_cost(X, Y, lambda_)

                self.b[str(j)] = np.array(b_try)
                self.b[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)

                grad_b[j-1][i] = (c2-c1) / (2*h)
            self.b[str(j)] = b_try
            
            W_try = np.copy(self.W[str(j)])
            for i in np.ndindex(self.W[str(j)].shape):
                self.W[str(j)] = np.array(W_try)
                self.W[str(j)][i] -= h
                c1, _ = self.compute_cost(X, Y, lambda_)
                
                self.W[str(j)] = np.array(W_try)
                self.W[str(j)][i] += h
                c2, _ = self.compute_cost(X, Y, lambda_)
                
                grad_W[j-1][i] = (c2-c1) / (2*h)
            self.W[str(j)] = W_try
        
        return grad_W[0], grad_b[0], grad_W[1], grad_b[1]

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
        _, P = self.evaluate_classifier(X)
        pre_l = np.matmul(Y.T, P)
        pre_l[pre_l == 0] = np.finfo(np.float32).eps
        l = -1/n * np.log(pre_l).trace()
        J = l + lambda_ * (np.sum(self.W['1']**2) + np.sum(self.W['2']**2))
        return J, l
    
    def mini_batch_GD(self, X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_, n_batch, n_epoch, n_step, eta_min, eta_max, shuffle, cost, cyclical):
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
        eta = eta_min
        t = 0
        l = 0
        rec = 2 * n_step / 10
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

                grad_W1, grad_b1, grad_W2, grad_b2 = self.compute_gradients(X_batch, Y_batch, lambda_)
                self.W['1'] -= eta * grad_W1
                self.b['1'] -= eta * grad_b1
                self.W['2'] -= eta * grad_W2
                self.b['2'] -= eta * grad_b2
                
                
                if cost & (t % rec == 0):
                    print(t)
                    cost_train, loss_train = self.compute_cost(X_train, Y_train, lambda_)
                    self.cost_train.append(cost_train)
                    self.loss_train.append(loss_train)
                    cost_val, loss_val = self.compute_cost(X_val, Y_val, lambda_)
                    self.cost_val.append(cost_val)
                    self.loss_val.append(loss_val)
                    acc_train = self.compute_accuracy(X_train, y_train)
                    acc_val = self.compute_accuracy(X_val, y_val)
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

    def compute_accuracy(self, X, y):
        """
        Parameters:
            X (d, n): image pixel data
            y (n): label for each image

        Returns:
            acc: accuracy of the model
        """
        n = X.shape[1]
        _, P = self.evaluate_classifier(X)
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
            X_train = np.concatenate((X_train, Xi[:,:9000]), axis=1)
            Y_train = np.concatenate((Y_train, Yi[:,:9000]), axis=1)
            y_train = y_train + yi[:9000]
            X_val = Xi[:,9000:]
            Y_val = Yi[:,9000:]
            y_val = yi[9000:]
    
    #X_val, Y_val, y_val = load_batch('datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test, y_test = load_batch('datasets/cifar-10-batches-py/test_batch')
    X_train_norm = normalization(X_train, X_train, norm)
    X_val_norm = normalization(X_val, X_train, norm)
    X_test_norm = normalization(X_test, X_train, norm)
    
    #test method
    model_NN = NeuralNetwork(n_hidden=50)
    model_NN.test_method(X_train, Y_train, lambda_=0, n_batch=20, ht=1e-3, type_norm=norm)
    model_NN.fit(X_train[:,:100], Y_train[:,:100], y_train[:100], X_val[:,:100], Y_val[:,:100], y_val[:100], lambda_=0, n_batch=100, n_epoch=200, n_step=0, eta_min=1e-2, eta_max=1e-2, shuffle=True, cost=True, cyclical=False, type_norm=norm)
    plt.plot(range(200), model_NN.cost_train)
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.title('Training Cost by Epoch')
    plt.savefig('epoch vs accuracy0.png')
    plt.close()
    
    n_batch=100
    eta_min=1e-5
    eta_max=1e-1

    # replicate figures
    n_epoch=[10,48]
    n_step=[500,800]
    lambda_=0.01
    for i in range(2):
        model_NN = NeuralNetwork(n_hidden=50)
        model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, n_batch=n_batch, n_epoch=n_epoch[i], n_step=n_step[i], eta_min=eta_min, eta_max=eta_max, shuffle=True, cost=True, cyclical=True, type_norm=norm)
        print('training accuracy {}: {}'.format(i, model_NN.predict(X_train_norm, y_train)))
        print('validation accuracy {}: {}'.format(i, model_NN.predict(X_val_norm, y_val)))
        print('testing accuracy {}: {}'.format(i, model_NN.predict(X_test_norm, y_test)))
        fname = "lambda:{}, n_batch:{}, n_epoch:{}, eta:{}, n_step:{}".format(lambda_, n_batch, n_epoch[i], 'cyclic', n_step[i])
        model_NN.plotting(filename=fname)
    
    
    # coarse-to-fine search
    # coarse search (2 cycles)
    n_epoch=8
    n_step=900
    l_min = -5
    l_max = -1
    lbd_list = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1]
    acc_val_list = []
    for l in lbd_list:
        lambda_=10**l
        model_NN = NeuralNetwork(n_hidden=50)
        model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, shuffle=True, cost=False, cyclical=True, type_norm=norm)
        acc = model_NN.predict(X_val_norm, y_val)
        acc_val_list.append(acc)
    print('all accuracy in coarse:')
    print(acc_val_list)
    top_l = np.asarray(lbd_list)[np.argsort(acc_val_list)[-3:][::-1]]
    print('top 3 lambda in coarse:')
    print(top_l)
    # fine search (3 cycles)
    l_min = np.min(top_l)
    l_max = np.max(top_l)
    n_epoch += 4
    lambda_dict = {}
    for i in range(3):
        lambda_list = []
        acc_val_list = []
        for _ in range(15):
            lbd = l_min + (l_max - l_min) * np.random.rand(1,1)[0][0]
            lambda_=10**lbd
            lambda_list.append(lambda_)
            model_NN = NeuralNetwork(n_hidden=50)
            model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, shuffle=True, cost=False, cyclical=True, type_norm=norm)
            acc = model_NN.predict(X_val_norm, y_val)
            acc_val_list.append(acc)
        plt.scatter(lambda_list, acc_val_list)
        plt.xlabel('Lambda')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy by Lambda [10^%.2f, 10^%.2f]' % (l_min, l_max))
        plt.savefig('lambda vs accuracy {}.png'.format(i))
        plt.close()
        lambda_dict[str(i)] = np.asarray(lambda_list)[np.argsort(acc_val_list)[-3:][::-1]]
        l_min = np.log10(np.min(lambda_dict[str(i)]))
        l_max = np.log10(np.max(lambda_dict[str(i)]))
    print('top 3 lambda in fine:')
    print(lambda_dict)
    with open('lambda_dict.p', 'wb') as fp:
        pickle.dump(lambda_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # get the best lambda (~3 cycle)
    lambda_ = 0.00197229
    n_epoch=12
    n_step=1000
    model_NN = NeuralNetwork(n_hidden=50)
    model_NN.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, lambda_=lambda_, n_batch=n_batch, n_epoch=n_epoch, n_step=n_step, eta_min=eta_min, eta_max=eta_max, shuffle=True, cost=True, cyclical=True, type_norm=norm)
    print('training accuracy {}: {}'.format(i, model_NN.predict(X_train_norm, y_train)))
    print('validation accuracy {}: {}'.format(i, model_NN.predict(X_val_norm, y_val)))
    print('testing accuracy {}: {}'.format(i, model_NN.predict(X_test_norm, y_test)))
    fname = "lambda:{}, n_batch:{}, n_epoch:{}, eta:{}, n_step:{}".format(lambda_, n_batch, n_epoch, 'cyclic', n_step)
    model_NN.plotting(filename=fname)
    
    end = time.time()
    print(end - start)
    