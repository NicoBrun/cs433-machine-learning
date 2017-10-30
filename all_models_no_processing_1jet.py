# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from implementations import logistic_regression,reg_logistic_regression, ridge_regression, least_squares, least_squares_GD, least_squares_SGD
from help_functions import calculate_loss,standardize
from proj1_helpers import load_csv_data,load_test_csv,predict_labels,create_csv_submission

""" process each column according to its distribution
    to transform it into a normal distribution and then
    standardize it to have nice ranges """
def data_processing(data_to_process, col_delete, col_sqrt, col_log, col_nothing_max,col_threshold ,col_nothing_norm, col_distance, col_pow_2, col_pow_3,col_pow_5, train = False, means = 0, stds = 0 ):

    data_processed = data_to_process

    #first we process the first col (-999 goes to mean)
    first_col = data_processed[:, 0]
    flag_col = np.zeros((len(first_col), 1))
    pos_value = first_col[first_col > 0]
    flag_col[first_col > 0] = 1
    first_col[first_col < 0] = np.mean(pos_value)

    first_col = np.reshape(first_col,(len(first_col),1))

    data_sqrt = data_processed[:,col_sqrt]
    data_sqrt[data_sqrt >=  0] = np.sqrt(data_sqrt[data_sqrt>=0])

    data_thresh = data_processed[:,col_threshold]

    data_thresh[:,0][data_thresh[:,0] > 0] = 1
    data_thresh[:,0][data_thresh[:,0] <= 0] = -1
    if(data_thresh.shape[1] > 1):

        data_thresh[:,1][data_thresh[:,1] > 0.5] = 1
        data_thresh[:,1][data_thresh[:,1] <= 0.5] = -1

    data_log = data_processed[:,col_log]
    data_log[data_log > 0] = np.log(data_log[data_log > 0])
    data_log[data_log == 0] = np.mean(data_log[data_log > 0])

    data_max = data_processed[:,col_nothing_max]
    max = np.amax(data_max,axis = 0)
    data_max /= max


    data_norm = data_processed[:,col_nothing_norm]

    columns_data_distance = []
    for col_distance_index in range(len(col_distance)):
        columns_data_distance.append(np.abs(data_processed[:,[col_distance[col_distance_index][0]]]-data_processed[:,[col_distance[col_distance_index][1]]]))
    data_distance = np.concatenate(columns_data_distance,axis = 1)

    data_pow_2 = data_processed[:,col_pow_2]**2

    data_pow_3 = data_processed[:,col_pow_3]**3

    data_pow_5 = data_processed[:,col_pow_5]**5

    data_to_standardize = np.concatenate((first_col, data_sqrt, data_log, data_norm, data_distance,data_pow_2,data_pow_3,data_pow_5),axis = 1)


    mean = means
    std = stds
    if(train) :
        mean = np.mean(data_to_standardize,axis = 0)
        std = np.std(data_to_standardize,axis = 0)

    data_to_standardize = standardize(data_to_standardize,mean,std)

    data_processed_standardized = np.concatenate((data_to_standardize,data_thresh,data_max,flag_col,np.ones((data_to_process.shape[0], 1))), axis=1)

    return data_processed_standardized, mean, std

""" separates data into train/test sets according to ratio """
def split_data(split, y_binary, input_data, seed = 1):
    np.random.seed(seed)
    index = np.arange(len(input_data))
    split = int(np.ceil(0.25*len(index)))
    np.random.shuffle(index)

    y_valid = y_binary[index[:split]]
    y_train = y_binary[index[split:]]

    x_valid = input_data[index[:split]]
    x_train = input_data[index[split:]]
    return y_valid, y_train, x_valid, x_train

""" set the values that are to -999 to the mean of the column"""
def data_arange(input_data):
    col_to_arange = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    new_data = input_data
    for i in col_to_arange:
        col = input_data[:, i]
        col[col == -999] = np.mean(col[col > -999])
        new_data[:, i] = col
    return new_data

""" determines the threshold for separating output
    that gives the smallest error """
def best_treshold(w, data_train, y_train):
    thresholds = np.linspace(-5,3,50)
    best_thresh = 0
    min_error = 1
    for thresh in thresholds :
        pred_thr = predict_labels(w,data_train,thresh)
        #resh = np.reshape(y_train, (len(y_train), 1)) - pred_thr
        err = np.count_nonzero(y_train[:, np.newaxis] - pred_thr)/len(y_train)
        if(err <= min_error):
            min_error = err
            best_thresh = thresh
    return best_thresh

""" computes the error from data and computed w"""
def global_error(y_valid, data_valid, y_train, data_train, w):
    loss_valid = calculate_loss(y_valid,data_valid,w)

    best_thresh = best_treshold(w, data_train, y_train)
    training = np.count_nonzero(
        predict_labels(w, data_train, best_thresh) - y_train[:, np.newaxis]) / len(y_train)
    pred = predict_labels(w,data_valid,best_thresh)
    nnz = np.count_nonzero(y_valid[:, np.newaxis]-pred) / len(y_valid)
    global_error = (len(y_valid)+len(y_train)) * nnz / len(y_binary)
    return global_error


data_path = "train.csv"
lambda_ = 0.001
max_iter = 3001
gamma = 0.00001
iter_step = 200

col_to_delete = [22]
col_log = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
col_sqrt = [0, 13, 16, 21, 23, 26, 29]
col_threshold = [11, 12]
col_nothing_max = [6, 14, 17, 24, 27]
col_nothing_norm = [7]
col_distance = [(15,18),(20,25),(18,28),(14,17),(15,25),(15,28),(18,20),(18,25),(18,28),(20,28)]
col_pow_2 = [3]
col_pow_3 = [19]
col_pow_5 = []

y_binary,input_data,ids = load_csv_data(data_path)

#Training part

cleaned_data = data_arange(input_data)
y_valid, y_train, data_valid, data_train = split_data(0.25, y_binary, cleaned_data, seed = 1)

#logistic regression
w_log, loss_train_log = logistic_regression(y_train,
                                            data_train,
                                            np.zeros((len(data_train[0]) ,1)),
                                            max_iter,
                                            gamma,
                                            data_valid,
                                            y_valid,
                                            iter_step)
global_error_log = global_error(y_valid, data_valid, y_train, data_train, w_log)
print("global error is {e}".format(e = global_error_log))

# regularized logistic regression
w_reg, loss_train_reg = reg_logistic_regression(y_train,
                                                data_train,
                                                0.05,
                                                np.zeros((len(data_train[0]) ,1)),
                                                max_iter,
                                                gamma,
                                                data_valid,
                                                y_valid,
                                                iter_step)
global_error_reg = global_error(y_valid, data_valid, y_train, data_train, w_reg)
print("global error is {e}".format(e = global_error_reg))

# ridge regression
print("ridge regression")
w_rid, loss_train_rid = ridge_regression(y_train, data_train, 0.05)
global_error_rid = global_error(y_valid, data_valid, y_train, data_train, w_rid)
print("global error is {e}".format(e = global_error_rid))


#Least squares logistic regression
print("Least squares logistic regression")
w_ls, loss_train_ls = least_squares(y_train, data_train)
global_error_ls = global_error(y_valid, data_valid, y_train, data_train, w_ls)
print("global error is {e}".format(e = global_error_ls))


#Least squares gradient descent
print("Least squares gradient descent")
w_lsg, loss_train_lsg = least_squares_GD(y_train, data_train, np.zeros((len(data_train[0]) ,1)),
                                                                        max_iter,
                                                                        gamma)
global_error_lsg = global_error(y_valid, data_valid, y_train, data_train, w_lsg)
print("global error is {e}".format(e = global_error_lsg))


#Least squares  stochastic gradient logistic regression
print("least squares stochastic gradient descent")
w_lss, loss_train_lss = least_squares_SGD(y_train, data_train, np.zeros((len(data_train[0]) ,1)),
                                                                        max_iter,
                                                                        gamma)
global_error_lss = global_error(y_valid, data_valid, y_train, data_train, w_lss)
print("global error is {e}".format(e = global_error_lss))
