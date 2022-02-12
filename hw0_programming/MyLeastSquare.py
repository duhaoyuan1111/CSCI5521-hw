
"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. final weight vector w
    2. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyLeastSquare(X,y) function.

"""

# Header
import numpy as np

# Solve the least square problem by setting gradients w.r.t. weights to zeros
def MyLeastSquare(X,y):
    # placeholders, ensure the function runs
    w = np.array([1.0,-1.0])
    error_rate = 1.0
    learning_rate = 0.01
    error = 0
    # calculate the optimal weights based on the solution of Question 1
    tran_x = np.transpose(X)
    for i in range(100):
        w = np.subtract(w, learning_rate * (2 * np.matmul(np.matmul(tran_x,X),w) - 2 * np.matmul(tran_x,y)))

    # compute the error rate
    temp = np.matmul(np.matmul(np.linalg.inv(np.matmul(tran_x,X)),tran_x),y)
    est_y = np.matmul(X,w)
    for i in range(len(est_y)):
        if est_y[i] >= 0:
            est_y[i] = 1
        else:
            est_y[i] = -1
    for i in range(len(est_y)):
        if est_y[i] != y[i]:
            error += 1
    error_rate = error / 100
    return (w,error_rate)
