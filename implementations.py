import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    pass
    #Linear regression using gradient descent

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    pass
    #Linear regression using stochastic gradient descent

def least_squares(y, tx):
    #Least squares regression using normal equations
    txt = tx.T
    w = np.linalg.inv(txt@tx)@txt@y
    return w

def ridge_regression(y, tx, lambda_ ):
    pass
    #Ridge regression using normal equations

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass
    #Logistic regression using gradient descent or SGD

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    pass
    #Regularized logistic regression using gradient descent or SGD
