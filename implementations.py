import numpy as np
from help_functions import compute_mse,learning_by_gradient_descent,learning_by_penalized_gradient,compute_gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    #Linear regression using gradient descent
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gr = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gr
    return w,loss

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

    return w,compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_ ):
    # Ridge regression using normal equations
    xtxli = np.dot(np.transpose(tx), tx) + (lambda_ * 2 * len(y)) * np.identity(len(tx[0]))
    w = np.dot(np.linalg.inv(xtxli), np.dot(np.transpose(tx), y))
    mse = compute_mse(y, tx, w)
    return  w,mse


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    #Logistic regression using gradient descent or SGD
    print("regression")
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iters):

        # get loss and update w.
        loss, w,grad = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if (iter % 1000== 0):
            print("step {i}, loss = {l}, gradient = {g}".format(i=iter,l = loss,g=np.linalg.norm(grad)))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w,loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    #Regularized logistic regression using gradient descent or SGD
    threshold = 1e-8
    losses = []
    w = initial_w
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w,loss