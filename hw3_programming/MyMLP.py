import numpy as np
import math

from numpy.core.fromnumeric import swapaxes

def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    if mean is not None:
        # directly use the mean and std precomputed from the training data
        data = np.divide(np.subtract(data,mean),std)
        return data
    else:
        # compute the mean and std based on the training data
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0) + 1e-15
        data = np.divide(np.subtract(data,mean),std)

        return data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for
    # training
    one_hot = np.zeros([len(label),10])
    for i in range (len(one_hot)):
        one_hot[i][label[i]] = 1

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    # You may receive some warning messages from Numpy. No worries, they should not affect your final results
    f_x = np.tanh(x)

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    f_x = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            beforeTanh_Z = train_x.dot(self.weight_1) + self.bias_1
            afterTanh_Z = tanh(beforeTanh_Z)
            beforeSM_Y = afterTanh_Z.dot(self.weight_2) + self.bias_2
            afterSM_Y = softmax(beforeSM_Y)

            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            delta = np.subtract(afterSM_Y, train_y)
            updateW2 = (afterTanh_Z.T).dot(delta)
            updateBias2 = np.sum(delta, axis=0, keepdims=True)
            delta2 = delta.dot(self.weight_2.T)*(1-np.power(afterTanh_Z, 2))
            updateW1 = np.dot(train_x.T, delta2)
            updateBias1 = np.sum(delta2, axis=0)
            updateW2 += lr*self.weight_2
            updateW1 += lr*self.weight_1

            #update the parameters based on sum of gradients for all training samples
            self.weight_1 += -lr*updateW1
            self.bias_1 += -lr*updateBias1
            self.weight_2 += -lr*updateW2
            self.bias_2 += -lr*updateBias2

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_accuracy = np.count_nonzero(predictions.reshape(-1) == valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_accuracy > best_valid_acc:
                best_valid_acc = valid_accuracy
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        beforeTanh_Z = x.dot(self.weight_1) + self.bias_1
        afterTanh_Z = tanh(beforeTanh_Z)
        beforeSM_Y = afterTanh_Z.dot(self.weight_2) + self.bias_2
        afterSM_Y = softmax(beforeSM_Y)

        # convert class probability to predicted labels
        y = np.zeros([len(x),]).astype('int')
        for i in range (len(afterSM_Y)):
            temp = np.where(afterSM_Y[i] == np.max(afterSM_Y[i]))[0].astype('int')
            y[i] = temp
        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        beforeTanh_Z = x.dot(self.weight_1) + self.bias_1
        afterTanh_Z = tanh(beforeTanh_Z)
        return afterTanh_Z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
