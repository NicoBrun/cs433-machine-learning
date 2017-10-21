# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from implementations import logistic_regression
from help_functions import calculate_loss,standardize
from proj1_helpers import load_csv_data,predict_labels

data_path = "train.csv"
seed = 1
lambda_ = 0.005
gamma = 0.00001
max_iter = 10000
number_feature = 30


print("début")

y_binary,input_data,ids = load_csv_data(data_path)


#separation along the 22th feature (number of jet) in order to do 4 models


indexes = [[], [], [], []]

for ind, item in enumerate(input_data):
    indexes[int(item[22])].append(ind)

#split 25/75 après on fera du k_fold mais c'est juste une toute première version 5-fold ?

global_error = 0

for i in range(0,4) :

    col_to_delete = [22]
    if(i == 0):
        col_to_delete = [4,5,6,12,22,23,24,25,26,27,28,29]
    elif (i == 1):
        col_to_delete = [4,5,6,12,22,26,27,28]

    np.random.seed(seed)
    index = indexes[i]
    split = int(np.ceil(0.25*len(index)))
    np.random.shuffle(index)

    y_valid = y_binary[index[:split]]
    y_train = y_binary[index[split:]]





    x_valid = np.delete(input_data[index[:split]],col_to_delete,1)
    # first column standardization, if neg = min(positive value)

    first_col = x_valid[:,0]
    flag_col = np.zeros((len(first_col), 1))
    pos_value_valid = first_col[ first_col> 0]
    flag_col[first_col> 0] = 1
    first_col[first_col< 0] = np.min(pos_value_valid)
    x_valid[:,0] = first_col

    mean = np.mean(x_valid,axis = 0)
    std =  np.std(x_valid,axis = 0)
    x_valid = standardize(x_valid,mean ,std)
    x_valid = np.concatenate((np.ones((len(y_valid),1)),x_valid,flag_col),axis = 1)



    x_train = np.delete(input_data[index[split:]],col_to_delete,1)

    first_col = x_train[:, 0]
    flag_col = np.zeros((len(first_col), 1))
    pos_value_train = first_col[first_col > 0]
    flag_col[first_col > 0] = 1
    first_col[first_col < 0] = np.min(pos_value_train)
    x_train[:, 0] = first_col

    x_train = standardize(x_train,mean,std)
    x_train = np.concatenate((np.ones((len(y_train), 1)), x_train,flag_col),axis = 1)

    #logistic regression
    w,loss_train = logistic_regression(y_train,x_train,np.zeros((number_feature + 2 - len(col_to_delete),1)),max_iter,gamma)
    print("end training")
    loss_valid = calculate_loss(y_valid,x_valid,w)
    pred = predict_labels(w,x_valid)

    nnz = np.count_nonzero(np.reshape(y_valid,(len(y_valid),1))-pred)

    error_rate = nnz/len(y_valid)
    global_error += (len(y_valid)+len(y_train)) * error_rate
    print("Pour jet {i} loss ={l} error_rate = {e}".format(i=i,l = loss_valid, e = error_rate))

global_error /= len(y_binary)

print("global error is {e}".format(e = global_error))
