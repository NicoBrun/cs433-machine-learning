import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y[:, np.newaxis] - tx @ w
    return (e * e).sum() / (2.0 * len(y))

def sigmoid(t):
    """apply sigmoid function on t."""
    empty = np.empty(t.shape)
    empty[t > 0] = 1/(1+ np.exp(-t[t>0]))
    empty[t <= 0 ] = np.exp(t[t<=0])/(1+np.exp(t[t<=0]))
    return empty

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    txw = np.matmul(tx, w)
    return np.logaddexp(0.0, txw).sum() - np.dot(y, txw)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-np.reshape(y,(len(y),1)))

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w_new = w - gamma*grad
    #grad is for debugging purpose
    return loss, w_new,grad

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    txw = tx.dot(w)
    diag = sigmoid(txw)*(np.ones(txw.shape)-sigmoid(txw))
    return np.matmul(np.multiply(tx,diag).T,tx)

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

    #on test avec Newton

    loss,gradient,_ = penalized_logistic_regression(y,tx,w,lambda_)

    w = w - gamma*gradient
    return loss, w,gradient

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y[:, np.newaxis] - tx @ w
    return tx.T @ e / (-len(y))

def standardize2(x,mean,std):
    ''' fill your code in here...
    '''
    mean_tile = np.tile(mean,[x.shape[0],1])
    std_dev = np.tile(std,[x.shape[0],1])
    x_new = x-mean_tile
    x_new = x_new/std_dev
    return x_new

def standardize(x,mean,std) :
    x_new = x-mean
    x_new = x_new/std
    return x_new

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        w = w - gamma*grad
        ws.append(w)
        losses.append(loss)

    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
