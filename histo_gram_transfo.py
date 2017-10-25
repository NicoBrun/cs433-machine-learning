import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data

data_path = "train.csv"
seed = 1
lambda_ = 0.001
gamma = 0.00001
max_iter = 1000
number_feature = 30

feature_to_watch = 29




y_binary,input_data,ids = load_csv_data(data_path)

indexes = [[], [], [], []]

for ind, item in enumerate(input_data):
    indexes[int(item[22])].append(ind)

for i in range(1,4) :

    data = input_data[indexes[i],feature_to_watch]
    if(feature_to_watch == 0) :
        data[data < 0] = np.mean(data[data>0])

    fig = plt.figure()
    fig.suptitle("jet {j} feature {f}".format(j=i,f = feature_to_watch))

    av = fig.add_subplot(231)
    av.hist(data,bins= 150)
    av.set_title("avant")

    norm = fig.add_subplot(232)
    norm.hist(((data - np.mean(data)) / np.std(data)), bins=150)
    norm.set_title("just normalization")


    log_=fig.add_subplot(233)
    data_ln = data
    data_ln[data > 0] = np.log(data[data > 0])
    data_ln[data < 0] = np.log(-data[data < 0])
    data_ln[data==0] = np.mean(data[data != 0])
    log_.hist((data_ln - np.mean(data_ln)) / np.std(data_ln), bins=150)
    log_.set_title("log")



    par= fig.add_subplot(234)

    data_pareto = data
    data_pareto[data > 0] = 1 / data_pareto[data > 0]
    data_pareto[data == 0] = np.mean(data[data > 0])

    data_pareto_normalised  = (data_pareto-np.mean(data_pareto))/np.std(data_pareto)

    par.hist(data_pareto_normalised, bins = 150)
    par.set_title("pareto")


    dec= fig.add_subplot(235)
    data_decal = data

    data_decal[data >= 0] = np.sqrt(data_decal[data >= 0])
    data_decal[data < 0] = np.sqrt(-data_decal[data < 0])

    dec.hist((data_decal - np.mean(data_decal)) / np.std(data_decal), bins=150)
    dec.set_title("decal")

    fig.tight_layout()

    fig.savefig("transformation/{f}_{i}.png".format(f=feature_to_watch,i=i))







