# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from implementations import logistic_regression,reg_logistic_regression
from help_functions import calculate_loss,standardize
from proj1_helpers import load_csv_data,load_test_csv,predict_labels,create_csv_submission
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

def data_arange(input_data):
    col_to_arange = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    new_data = input_data
    for i in col_to_arange:
        col = input_data[:, i]
        col[col == -999] = np.mean(col[col > -999])
        new_data[:, i] = col
    return new_data

def best_treshold(w, data_train, y_train):
    thresholds = np.linspace(-5,3,200)
    best_thresh = 0
    min_error = 1
    for thresh in thresholds :
        pred_thr = predict_labels(w,data_train,thresh)
        err =np.count_nonzero(np.reshape(y_train, (len(y_train), 1)) - pred_thr)/len(y_train)
        if(err <= min_error):
            min_error = err
            best_thresh = thresh
    return best_thresh

def create_figure(iter_val_errors, iter_train_errors, max_iters, iter_step):

    iter_step = 200 #to plot validation and training error

    fig = plt.figure()
    st = fig.suptitle("Train and validation error")
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, max_iter, num = np.ceil(max_iter/iter_step)), iter_val_errors, 'b', label = 'v')
    ax.plot(np.linspace(0, max_iter, num = np.ceil(max_iter / iter_step)), iter_train_errors, 'g', label='t')
    ax.legend(loc='upper right')
    fig.tight_layout()

    st.set_y(0.95)
    fig.subplots_adjust(top = 0.85)

    fig.savefig("valid_train_error_with_thresh_no_jet.png")




data_path = "train.csv"
lambda_ = 0.001
max_iter = 30001
gamma = 0.00001
iter_step = 200

number_feature = 30

feature_to_watch = 7

col_to_delete = [22]  # almost constants values
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

#split 25/75 après on fera du k_fold mais c'est juste une toute première version 5-fold ?

#Training part
global_error = 0

"""np.random.seed(seed)
index = np.arange(len(input_data))
split = int(np.ceil(0.25*len(index)))
np.random.shuffle(index)

y_valid = y_binary[index[:split]]
y_train = y_binary[index[split:]]

x_valid = input_data[index[:split]]
x_train = input_data[index[split:]]"""

cleaned_data = data_arange(input_data)
y_valid, y_train, x_valid, x_train = split_data(0.25, y_binary, cleaned_data, seed = 1)



data_train,mean,std = data_processing(x_train, col_to_delete, col_sqrt,col_log,col_nothing_max,col_threshold,col_nothing_norm, col_distance,col_pow_2,col_pow_3,col_pow_5, train = True)
data_valid,_,_ = data_processing(x_valid, col_to_delete, col_sqrt, col_log,col_nothing_max,col_threshold,col_nothing_norm, col_distance,col_pow_2,col_pow_3,col_pow_5, train = False, means = mean, stds = std)


#logistic regression
# 3 = bias, 1st column, flag column
print("no jet")
w, loss_train, iter_val_errors, iter_train_errors = logistic_regression(y_train,
                                                                        data_train,
                                                                        np.zeros((3+len(col_sqrt)
                                                                            +len(col_log)
                                                                            +len(col_nothing_max)
                                                                            +len(col_threshold)
                                                                            +len(col_nothing_norm)
                                                                            +len(col_distance)
                                                                            +len(col_pow_2)
                                                                            +len(col_pow_3)
                                                                            +len(col_pow_5) ,1)),
                                                                        max_iter,
                                                                        gamma,
                                                                        data_valid,
                                                                        y_valid,
                                                                        iter_step)

create_figure(iter_val_errors, iter_train_errors, max_iter, iter_step)

loss_valid = calculate_loss(y_valid,data_valid,w)

best_thresh = best_treshold(w, data_train, y_train)

training_error = np.count_nonzero(
    predict_labels(w, data_train, best_thresh) - np.reshape(y_train, (len(y_train), 1))) / len(y_train)

pred = predict_labels(w,data_valid,best_thresh)

nnz = np.count_nonzero(np.reshape(y_valid,(len(y_valid),1))-pred) / len(y_valid)

global_error += (len(y_valid)+len(y_train)) * nnz / len(y_binary)

print("best treshold is {t}".format(t = best_thresh))
print("global error is {e}".format(e = global_error))



#Testing part
input_test, ids = load_test_csv("test.csv")

#features processing

indexes_test = [[], [], [], []]

sols = []
x_test = data_arange(input_test)

#process the first column with adding a flag
data_test,_,_ = data_processing(x_test,col_to_delete,col_sqrt,col_log,col_nothing_max,col_threshold,col_nothing_norm, col_distance,col_pow_2,col_pow_3,col_pow_5,train= False, means= mean, stds = std)

#prediction

y_test = predict_labels(w, data_test, best_thresh)

y_test[y_test == 0] = -1

sol = np.concatenate((y_test,np.reshape(ids,(len(y_test),1))), axis = 1)

create_csv_submission(sol[:,1],sol[:,0],"no_jet_with_thresh.csv")
