import numpy as np
from proj1_helpers import predict_labels


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

""" Same functions as in implementation.py, but returns validation and
    train error to have nice plots """
def logistic_regression(y, tx, initial_w, max_iters, gamma, tx_valid, y_valid,iter_step):
    #Logistic regression using gradient descent or SGD
    threshold = 1e-8
    losses = []
    w = initial_w
    gam = gamma
    val_errors = []
    train_errors = []
    for iter in range(max_iters):

        # get loss and update w.
        loss, w,grad = learning_by_gradient_descent(y, tx, w, gam)
        prev_grad = grad
        # converge criterion
        losses.append(loss)

        if(iter % iter_step == 0) :
            val_errors.append(np.count_nonzero(predict_labels(w,tx_valid,0.5) - np.reshape(y_valid,(len(y_valid),1)) )/len(y_valid))
            train_errors.append(np.count_nonzero(predict_labels(w,tx,0.5) - np.reshape(y,(len(y),1)) )/len(y))

        if (iter % 1000== 0):
            print("step {i}, loss = {l}, gradient = {g}".format(i=iter,l = loss,g=np.linalg.norm(grad)))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w,loss, val_errors, train_errors

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma,tx_valid,y_valid,iter_step):
    #Regularized logistic regression using gradient descent or SGD
    threshold = 1e-8
    losses = []
    w = initial_w
    val_errors = []
    train_errors = []

    for iter in range(max_iters):
        # get loss and update w.
        loss, w,grad = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)

        if (iter % iter_step == 0):
            val_errors.append(np.count_nonzero(predict_labels(w, tx_valid, 0.5) - np.reshape(y_valid, (len(y_valid), 1))) / len(y_valid))
            train_errors.append(np.count_nonzero(predict_labels(w, tx, 0.5) - np.reshape(y, (len(y), 1))) / len(y))

        if (iter % 1000== 0):
            print("step {i}, loss = {l}, gradient = {g}".format(i=iter,l = loss,g=np.linalg.norm(grad)))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w,loss, val_errors, train_errors
