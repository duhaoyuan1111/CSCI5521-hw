"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. number of iterations / passes it takes until your weight vector stops changing
    2. final weight vector w
    3. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyPerceptron(X,y,w) function.

"""
# Hints
# one can use numpy package to take advantage of vectorization
# matrix multiplication can be done via nested for loops or
# matmul function in numpy package


# Header
import numpy as np

# Implement the Perceptron algorithm
def MyPerceptron(X,y,w0=[1.0,-1.0]):
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w=w0
    error_rate = 1.00
    new_w = []
    # loop until convergence (w does not change at all over one pass)
    # or until max iterations are reached
    # (current pass w ! = previous pass w), then do:
    #
    misclassified = 0
        # for each training sample (x,y):
        # if actual target y does not match the predicted target value, update the weights
    for t in range(0,100):
        if (y[t]*(w[0]*X[t][0]+w[1]*X[t][1])) <= 0:
            new_w.append(w[0] + X[t][0]*y[t])
            new_w.append(w[1] + X[t][1]*y[t])
            
            # calculate the number of iterations as the number of updates
            misclassified+=1
            k = t
            if new_w[0] == w[0] and new_w[1] == w[1]:
                break
            else:
                w[0] = new_w[0]
                w[1] = new_w[1]
                new_w = []
        
    
    # make prediction on the csv dataset using the feature set
    # Note that you need to convert the raw predictions into binary predictions using threshold 0
    
    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples
    error_rate = misclassified / 100

    return (w, k, error_rate)
