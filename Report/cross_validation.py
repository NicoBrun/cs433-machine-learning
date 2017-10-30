import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from help_functions import logistic_regression,reg_logistic_regression

#function provided in lab4
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def data_arange(input_data):
    col_to_arange = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    new_data = input_data
    for i in col_to_arange:
        col = input_data[:, i]
        col[col == -999] = np.mean(col[col > -999])
        new_data[:, i] = col
    return new_data

#function provided in lab4
def cross_validation_visualization(gammas, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(gammas, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(gammas, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

data_path = "train.csv"
y_binary,input_data,ids = load_csv_data(data_path)

seed = 1
#gamma = 0.00001
list_gamma = np.logspace(-5, 0, 15)
max_iters = 2001
iter_step = 100
k_fold = 4

data = data_arange(input_data)

k_indices = build_k_indices(y_binary,k_fold,seed)

list_val_errors = []
list_train_errors = []
for gamma in list_gamma:
    tmp_list_val_error = []
    tmp_list_train_error = []
    print("gamma:"+str(gamma))
    for k in range(k_fold):
        y_valid = y_binary[k_indices[k]]
        #train_indices = k_indices[(k+1)%4]+k_indices[(k+2)%4]+k_indices[(k+3)%4].reshape(-1)
        train_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)

        y_train = y_binary[train_indices]

        x_valid = data[k_indices[k]]
        x_train = data[train_indices]

        w, loss_train, val_errors, train_errors = logistic_regression(y_train, x_train, np.zeros((30 ,1)), max_iters, gamma, x_valid, y_valid, iter_step)
        tmp_list_val_error.append(val_errors)
        tmp_list_train_error.append(train_errors)
        mean_val_errors = np.mean(tmp_list_val_error)
        mean_train_errors = np.mean(tmp_list_train_error)

    list_val_errors.append(mean_val_errors)
    list_train_errors.append(mean_train_errors)
cross_validation_visualization(list_gamma, list_val_errors, list_train_errors)
