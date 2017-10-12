import numpy as np

## pour le moment helper pour faire fonctionner les autres
def MSE(y,tx,w) :
    e = y-np.dot(tx,w)
    e = np.square(e)
    mse = np.sum(e)
    mse = mse/len(y)
    return mse

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    pass
    #Linear regression using gradient descent

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    pass
    #Linear regression using stochastic gradient descent

def least_squares(y, tx):
    #Least squares regression using normal equations
    xtx = np.dot(np.transpose(tx), tx)

    rank = np.linalg.matrix_rank(xtx)

    if (rank < len(xtx[0])):
        # non invertible matrix
        w = np.dot(np.linalg.pinv(tx), y)
    else:
        # invertible matrix
        w = np.dot(np.linalg.inv(xtx), np.dot(np.transpose(tx), y))

    return MSE(y, tx, w), w


def ridge_regression(y, tx, lambda_ ):
    # Ridge regression using normal equations
    xtxli = np.dot(np.transpose(tx), tx) + (lambda_ * 2 * len(y)) * np.identity(len(tx[0]))
    w = np.dot(np.linalg.inv(xtxli), np.dot(np.transpose(tx), y))
    mse = MSE(y, tx, w)
    return mse, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass
    #Logistic regression using gradient descent or SGD

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    pass
    #Regularized logistic regression using gradient descent or SGD