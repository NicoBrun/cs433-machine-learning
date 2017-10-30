
import numpy as np
import matplotlib.pyplot as plt
from implementations import logistic_regression,reg_logistic_regression, ridge_regression, least_squares, least_squares_GD, least_squares_SGD
from help_functions import calculate_loss,standardize
from proj1_helpers import load_csv_data,load_test_csv,predict_labels,create_csv_submission

data_path = "train.csv"
name_error_image = "valid_train_error_with_thresh.png"
seed = 1
lambda_ = 0.001
gamma = 0.00001
max_iter = 3001
iter_step = 200 #to plot validation and training error

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
            print(sols[0].shape)
            sols[0] = np.concatenate((sols[0],sol),axis = 0)

    return sols

""" determines the threshold for separating output
    that gives the smallest error """
def best_treshold(w, data_train, y_train):
    thresholds = np.linspace(-5,3,50)
    best_thresh = 0
    min_error = 1
    for thresh in thresholds :
        pred_thr = predict_labels(w,data_train,thresh)
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
    global_error = (len(y_valid)+len(y_train)) * nnz
    return global_error

print("début")

#load data and separate it into 4 according to their jet
y_binary,input_data,ids = load_csv_data(data_path)
indexes = separate_from_jet(input_data)


ws_log = []
ws_reg = []
ws_ls = []
ws_lsg = []
ws_lss = []
ws_rid = []

means = []
stds = []

global_error_log = 0
global_error_reg = 0
global_error_ls = 0
global_error_lsg = 0
global_error_lss = 0
global_error_rid  = 0

for i in range(4):
    #train/test ratio is 0.75/0.25 for now
    y_valid, y_train, x_valid, x_train = split_data(0.25, y_binary, input_data, indexes[i])

    # process data
    col_to_delete, col_log, col_sqrt, col_threshold, col_nothing_max, col_nothing_norm, col_distance, col_pow_2, col_pow_3, col_pow_5 = get_columns(i)
    #data_train, mean, std = data_processing(x_train, i, train = True)
    #means.append(mean)
    #stds.append(std)
    #data_valid, _, _ = data_processing(x_valid, i, train = False, means = mean, stds = std)
    data_train = x_train
    data_valid = x_valid
    #logistic regression
    w_log,loss_train_log = logistic_regression(y_train,
                                            data_train,
                                            np.zeros((len(data_train[0]) ,1)),
                                            max_iter,
                                            gamma,
                                            data_valid,
                                            y_valid,
                                            iter_step)
    ws_log.append(w_log)
    global_error_log += global_error(y_valid, data_valid, y_train, data_train, w_log) #est-ce vraiment juste de compter le train ?
    print("error for log and jet {i} is {e}".format(i = i, e = global_error_log))

    # regularized logistic regression
    w_reg, loss_train_reg, = reg_logistic_regression(y_train, data_train, 0.05,
                                                        np.zeros((len(data_train[0]) ,1)),
                                                        max_iter,
                                                        gamma,
                                                        data_valid,
                                                        y_valid,
                                                        iter_step)

    global_error_reg += global_error(y_valid, data_valid, y_train, data_train, w_reg)
    print("global error for regression and jet {i} is {e}".format(i = i, e = global_error_reg))

    # ridge regression
    print("ridge regression")
    w_rid, loss_train_rid = ridge_regression(y_train, data_train, 0.05)
    global_error_rid += global_error(y_valid, data_valid, y_train, data_train, w_rid)
    print("global error for ridge and jet {i} is {e}".format(i = i, e = global_error_rid))

    #Least squares logistic regression
    print("Least squares logistic regression")
    w_ls, loss_train_ls = least_squares(y_train, data_train)
    global_error_ls += global_error(y_valid, data_valid, y_train, data_train, w_ls)
    print("global error for least squares and jet {i} is {e}".format(i = i, e = global_error_ls))


    #Least squares gradient descent
    print("Least squares gradient descent")
    w_lsg, loss_train_lsg = least_squares_GD(y_train, data_train, np.zeros((len(data_train[0]) ,1)),
                                                                            max_iter,
                                                                            gamma)
    global_error_lsg += global_error(y_valid, data_valid, y_train, data_train, w_lsg)
    print("global error for least squares and jet {i} is {e}".format(i = i, e = global_error_lsg))

    #Least squares gradient descent
    print("Least squares stochastic gradient descent")
    w_lss, loss_train_lss = least_squares_SGD(y_train, data_train, np.zeros((len(data_train[0]) ,1)),
                                                                            max_iter,
                                                                            gamma)
    global_error_lss += global_error(y_valid, data_valid, y_train, data_train, w_lss)
    print("global error for stochastic least squares and jet {i} is {e}".format(i = i, e = global_error_lss))



global_error_log /= len(y_binary)
global_error_reg /= len(y_binary)
global_error_rid /= len(y_binary)
global_error_ls /= len(y_binary)
global_error_lsg /= len(y_binary)
global_error_lss /= len(y_binary)

print("global error for log is {e}".format(e = global_error_log))
print("global error for regression is {e}".format(e = global_error_reg))
print("global error for ridge is {e}".format(e = global_error_rid))
print("global error for least squares is {e}".format(e = global_error_ls))
print("global error for least squares gradient descent is {e}".format(e = global_error_lsg))
print("global error for stochastic least squares is {e}".format(e = global_error_lss))
