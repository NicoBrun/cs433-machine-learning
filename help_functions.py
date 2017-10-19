import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.T.dot(e) / (2 * len(e))
    return mse

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(np.ones(t.shape)+np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    txtw = tx.dot(w)
    loss = -np.multiply(y,txtw) + np.log(1 + np.exp(txtw))
    return np.sum(loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w -gamma*grad
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    txw = tx.dot(w)
    diag = sigmoid(txw)*(np.ones(txw.shape)-sigmoid(txw))
    S = np.diag(np.ndarray.flatten(diag))
    return np.dot(tx.T,np.dot(S,tx))

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    penality = lambda_*np.linalg.norm(w)**2
    diag = np.diag(np.repeat(2*lambda_, len(w)))
    return calculate_loss(y,tx,w) + penality, calculate_gradient(y,tx,w) + lambda_*2*w, calculate_hessian(y,tx,w) + diag

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    loss,gradient,_ = penalized_logistic_regression(y,tx,w,lambda_)

    w = w - lambda_*gradient
    return loss, w

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y- np.dot(tx,w)
    return -np.dot(np.transpose(tx),e) /len(y)