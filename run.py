# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from help_functions import calculate_loss, standardize
from implementations import logistic_regression
from proj1_helpers import load_csv_data, load_test_csv, predict_labels, create_csv_submission

data_path = "train.csv"
name_error_image = "valid_train_error_with_thresh.png"
seed = 1

# lambda and gamma were determined by cross validation
lambda_ = 0.00001
gamma = 0.00001

max_iter = 301

""" returns the columns according to what operations have
    to be done on them in order to get the best model. operations
    depend on the jet """
def get_columns(i):
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
    if (i == 0):
        col_to_delete = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
        col_log = [0, 1, 2, 3, 8, 9, 10, 13, 16, 19, 21]
        col_sqrt = [0, 13, 16, 21]
        col_threshold = [11]
        col_nothing_max = [14, 17]
        col_nothing_norm = [7]
        col_distance = [(15,18),(14,17),(18,20)]
        col_pow_2 = []
        col_pow_3 = []
        col_pow_5 = []
        #20

    elif (i == 1):
        col_to_delete = [4, 5, 6, 12, 22, 26, 27, 28]
        col_log = [0, 1, 2, 3, 8, 9, 10, 13, 16, 19, 21, 23, 29]
        col_sqrt = [0, 13, 16, 21, 23, 29]
        col_threshold = [11]
        col_nothing_max = [14, 17, 24]
        col_nothing_norm = [7]
        col_distance = [(15,18),(20,25),(14,17),(15,25),(18,20),(18,25)]
        col_pow_2 = [3]
        col_pow_3 = [19]
        col_pow_5 = []

    elif (i == 3):
        col_pow_2 = []
        col_pow_3 = [8, 19]
        col_pow_5 = [3]

    return col_to_delete, col_log, col_sqrt, col_threshold, col_nothing_max, col_nothing_norm, col_distance, col_pow_2, col_pow_3, col_pow_5

""" process each column according to its distribution
    to transform it into a normal distribution and then
    standardize it to have nice ranges """
def data_processing(data_to_process, jet, train = False, means = 0, stds = 0 ):

    data_processed = data_to_process

    col_to_delete, col_log, col_sqrt, col_threshold, col_nothing_max, col_nothing_norm, col_distance, col_pow_2, col_pow_3, col_pow_5 = get_columns(jet)

    #set first column values to mean where it was -999
    first_col = data_processed[:, 0]
    flag_col = np.zeros((len(first_col), 1))
    pos_value = first_col[first_col > 0]
    flag_col[first_col > 0] = 1
    first_col[first_col < 0] = np.mean(pos_value)

    first_col = np.reshape(first_col,(len(first_col),1))

    # apply square root to corresponding columns
    data_sqrt = data_processed[:,col_sqrt]
    data_sqrt[data_sqrt >=  0] = np.sqrt(data_sqrt[data_sqrt >= 0])

    #separate corresponding columns according to a treshold of 0
    data_thresh = data_processed[:,col_threshold]
    data_thresh[:,0][data_thresh[:,0] > 0] = 1
    data_thresh[:,0][data_thresh[:,0] <= 0] = -1
    if(data_thresh.shape[1] > 1):
        data_thresh[:,1][data_thresh[:,1] > 0.5] = 1
        data_thresh[:,1][data_thresh[:,1] <= 0.5] = -1

    # apply log to corresponding columns
    data_log = data_processed[:,col_log]
    data_log[data_log > 0] = np.log(data_log[data_log > 0])
    data_log[data_log == 0] = np.mean(data_log[data_log > 0])

    # divide by max to get in a [0, 1] range
    data_max = data_processed[:,col_nothing_max]
    max = np.amax(data_max,axis = 0)
    data_max /= max

    # get the columns where there are no operations to do
    data_norm = data_processed[:,col_nothing_norm]

    # process features that go together
    columns_data_distance = []
    for col_distance_index in range(len(col_distance)):
        columns_data_distance.append(np.abs(data_processed[:,[col_distance[col_distance_index][0]]]-data_processed[:,[col_distance[col_distance_index][1]]]))
    data_distance = np.concatenate(columns_data_distance,axis = 1)

    # apply power
    data_pow_2 = data_processed[:,col_pow_2]**2

    data_pow_3 = data_processed[:,col_pow_3]**3

    data_pow_5 = data_processed[:,col_pow_5]**5

    # put new columns together
    data_to_standardize = np.concatenate((first_col, data_sqrt, data_log, data_norm, data_distance,data_pow_2,data_pow_3,data_pow_5),axis = 1)

    # standardize everything to have nice input data
    mean = means
    std = stds
    if(train) :
        mean = np.mean(data_to_standardize,axis = 0)
        std = np.std(data_to_standardize,axis = 0)

    data_to_standardize = standardize(data_to_standardize,mean,std)

    data_processed_standardized = np.concatenate((data_to_standardize,data_thresh,data_max,flag_col,np.ones((data_to_process.shape[0], 1))), axis=1)

    return data_processed_standardized, mean, std

""" returns an array of 4 datas sets splitted according to their jet """
def separate_from_jet(data):
    indexes = [[], [], [], []]
    for ind, item in enumerate(data):
        indexes[int(item[22])].append(ind)
    return indexes

""" separates data into train/test sets according to ratio """
def split_data(ratio, y_binary, input_data, index, seed = 1):
    np.random.seed(seed)
    #index = np.arange(len(input_data))
    split = int(np.ceil(ratio*len(index)))
    np.random.shuffle(index)

    y_valid = y_binary[index[:split]]
    y_train = y_binary[index[split:]]

    x_valid = input_data[index[:split]]
    x_train = input_data[index[split:]]

    return y_valid, y_train, x_valid, x_train

""" returns the predicted y datas according to jet """
def prediction_solutions(test_path, ws, means, stds):
    input_test, ids = load_test_csv(test_path)

    #features processing

    indexes_test = separate_from_jet(input_test)
    sols = []

    for i in range(0,4):
        x_test = input_test[indexes_test[i]]

        #process the first column with adding a flag
        data_test, _, _ = data_processing(x_test, i, train= False, means= means[i], stds = stds[i])

        #prediction
        y_test = predict_labels(ws[i], data_test, threshes[i])
        y_test[y_test == 0] = -1

        sol = np.concatenate((y_test,np.reshape(ids[indexes_test[i]],(len(y_test),1))), axis = 1)

        if(i == 0):
            sols.append(sol)
        else :
            sols[0] = np.concatenate((sols[0],sol),axis = 0)

    return sols

""" determines the threshold for separating output
    that gives the smallest error """
def best_threshold(w, data_train, y_train):
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

print("start")

#load data and separate it into 4 according to their jet
y_binary,input_data,ids = load_csv_data(data_path)
indexes = separate_from_jet(input_data)

ws = []
means = []
stds = []

global_error = 0
threshes = []

# training part
for i in range(4):
    #data processing
    col_to_delete, col_log, col_sqrt, col_threshold, col_nothing_max, col_nothing_norm, col_distance, col_pow_2, col_pow_3, col_pow_5 = get_columns(i)
    y_valid, y_train, x_valid, x_train = split_data(0.25, y_binary, input_data, indexes[i])

    data_train, mean, std = data_processing(x_train, i, train = True)
    means.append(mean)
    stds.append(std)
    data_valid, _, _ = data_processing(x_valid, i, train = False, means = mean, stds = std)

    #logistic regression
    w,loss_train = logistic_regression(y_train,
                                    data_train,
                                    np.zeros((3+len(col_sqrt)+
                                        len(col_log)+
                                        len(col_nothing_max)+
                                        len(col_threshold)+
                                        len(col_nothing_norm)+
                                        len(col_distance)+
                                        len(col_pow_2)+
                                        len(col_pow_3)+
                                        len(col_pow_5) ,1)),
                                    max_iter,
                                    gamma,
                                    data_valid,
                                    y_valid)
    ws.append(w)
    print("end training")

    loss_valid = calculate_loss(y_valid,data_valid,w)

    #separates output
    best_thresh = best_threshold(w, data_train, y_train)
    threshes.append(best_thresh)
    print("for jet {i} the best thresh is {t}".format(i=i,t=best_thresh))

    #computes the error for a jet
    training_error = np.count_nonzero(
        predict_labels(w, data_train, best_thresh) - np.reshape(y_train, (len(y_train), 1))) / len(y_train)
    pred = predict_labels(w,data_valid,best_thresh)
    nnz = np.count_nonzero(np.reshape(y_valid,(len(y_valid),1))-pred)
    validation_error = nnz / len(y_valid)
    global_error += (len(y_valid)+len(y_train)) * validation_error

    print("For jet {i} loss ={l} validation_error = {e} and training_error = {t}".format(i=i, l = loss_valid, e = validation_error,t=training_error))

global_error /= len(y_binary)
print("global error is {e}".format(e = global_error))

# prediction part
sols = prediction_solutions("test.csv", ws, means, std)
create_csv_submission(sols[0][:,1],sols[0][:,0],"4_models_with_thresh.csv")
