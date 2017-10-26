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
    data_sqrt[data_sqrt >=  0] = np.sqrt(data_sqrt[data_sqrt>0])

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



data_path = "train.csv"
seed = 1
lambda_ = 0.001
gamma = 0.00001
max_iter = 22001
number_feature = 30

feature_to_watch = 7


print("début")

y_binary,input_data,ids = load_csv_data(data_path)


#separation along the 22th feature (number of jet) in order to do 4 models


indexes = [[], [], [], []]

for ind, item in enumerate(input_data):
    indexes[int(item[22])].append(ind)

ws = []
means = []
stds = []

#split 25/75 après on fera du k_fold mais c'est juste une toute première version 5-fold ?

global_error = 0

for i in range(0,4):

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
        col_nothing_max = [6, 14, 17]
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
        col_nothing_max = [6, 14, 17, 24]
        col_nothing_norm = [7]
        col_distance = [(15,18),(20,25),(14,17),(15,25),(18,20),(18,25)]
        col_pow_2 = [3]
        col_pow_3 = [19]
        col_pow_5 = []

    elif (i == 3):
        col_pow_2 = []
        col_pow_3 = [8, 19]
        col_pow_5 = [3]



    np.random.seed(seed)
    index = indexes[i]
    split = int(np.ceil(0.25*len(index)))
    np.random.shuffle(index)

    y_valid = y_binary[index[:split]]
    y_train = y_binary[index[split:]]


    x_train = input_data[index[split:]]

    data_train,mean,std = data_processing(x_train, col_to_delete, col_sqrt,col_log,col_nothing_max,col_threshold,col_nothing_norm, col_distance,col_pow_2,col_pow_3,col_pow_5, train = True)

    means.append(mean)
    stds.append(std)
    x_valid = input_data[index[:split]]
    data_valid,_,_ = data_processing(x_valid, col_to_delete, col_sqrt, col_log,col_nothing_max,col_threshold,col_nothing_norm, col_distance,col_pow_2,col_pow_3,col_pow_5, train = False, means = mean, stds = std)


    #logistic regression
    w,loss_train = logistic_regression(y_train,data_train,np.zeros((3+len(col_sqrt)+len(col_log)+len(col_nothing_max)+len(col_threshold)+len(col_nothing_norm)+len(col_distance)+len(col_pow_2)+len(col_pow_3)+len(col_pow_5) ,1)),max_iter,gamma)
    ws.append(w)
    print("end training")
    loss_valid = calculate_loss(y_valid,data_valid,w)
    pred = predict_labels(w,data_valid)

    nnz = np.count_nonzero(np.reshape(y_valid,(len(y_valid),1))-pred)

    error_rate = nnz/len(y_valid)
    global_error += (len(y_valid)+len(y_train)) * error_rate
    print("For jet {i} loss ={l} error_rate = {e}".format(i=i,l = loss_valid, e = error_rate))

global_error /= len(y_binary)

print("global error is {e}".format(e = global_error))


input_test, ids = load_test_csv("test.csv")

#features processing

indexes_test = [[], [], [], []]

sols = []
for ind, item in enumerate(input_test):
    indexes_test[int(item[22])].append(ind)

for i in range(0,4):

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
        col_nothing_max = [6, 14, 17]
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
        col_nothing_max = [6, 14, 17, 24]
        col_nothing_norm = [7]
        col_distance = [(15,18),(20,25),(14,17),(15,25),(18,20),(18,25)]
        col_pow_2 = [3]
        col_pow_3 = [19]
        col_pow_5 = []

    elif (i == 3):
        col_pow_2 = []
        col_pow_3 = [8, 19]
        col_pow_5 = [3]



    x_test = input_test[indexes_test[i]]

    #process the first column wuth adding a flag
    data_test,_,_ = data_processing(x_test,col_to_delete,col_sqrt,col_log,col_nothing_max,col_threshold,col_nothing_norm, col_distance,col_pow_2,col_pow_3,col_pow_5,train= False, means= means[i], stds = stds[i])

    #prediction

    y_test = predict_labels(ws[i],data_test)

    y_test[y_test == 0] = -1



    sol = np.concatenate((y_test,np.reshape(ids[indexes_test[i]],(len(y_test),1))), axis = 1)

    if(i == 0):
        sols.append(sol)
    else :
        print(sols[0].shape)
        sols[0] = np.concatenate((sols[0],sol),axis = 0)



create_csv_submission(sols[0][:,1],sols[0][:,0],"yolo_4_models_data_processing.csv")
