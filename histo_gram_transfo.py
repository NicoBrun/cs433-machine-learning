import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data

data_path = "train.csv"
seed = 1
lambda_ = 0.001
gamma = 0.00001
max_iter = 1000
number_feature = 30

#feature_to_watch = 29




y_binary,input_data,ids = load_csv_data(data_path)

indexes = [[], [], [], []]

for ind, item in enumerate(input_data):
    indexes[int(item[22])].append(ind)

for feature_to_watch in range(30):
    for i in range(1,4) :

        if (i == 0) and feature_to_watch in [4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29]:
            continue
        elif (i == 1) and feature_to_watch in [4, 5, 6, 12, 26, 27, 28, 29]:
            continue
        elif feature_to_watch != 22:
            output = y_binary[indexes[i]]
            data = input_data[indexes[i],feature_to_watch]
            data1 = data[output == 1]
            data2 = data[output == 0]
            if(feature_to_watch == 0) :
                data[data < 0] = np.mean(data[data>0])

            fig = plt.figure()
            st = fig.suptitle("jet {j} feature {f}".format(j=i,f = feature_to_watch))

            av = fig.add_subplot(231)
            #av.hist((data1, data2),bins= 150, color = ['r', 'b'])
            av.hist(data1, bins = 150, color = 'b', label='s')
            av.hist(data2, bins = 150, color = 'g', label='b')
            av.legend(loc='upper right')
            av.set_title("Initial data")

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

            data_square = data
            #data_pareto[data > 0] = 1 / data_pareto[data > 0]
            data_square = np.square(data_square)
            data_square[data == 0] = np.mean(data[data > 0])

            data_square_normalised  = (data_square - np.mean(data_square)) / np.std(data_square)

            par.hist(data_square_normalised, bins = 150)
            par.set_title("x^2")


            dec= fig.add_subplot(235)
            data_decal = data

            data_decal[data >= 0] = np.sqrt(data_decal[data >= 0])
            data_decal[data < 0] = np.sqrt(-data_decal[data < 0])

            dec.hist((data_decal - np.mean(data_decal)) / np.std(data_decal), bins=150)
            dec.set_title("sqrt")


            cub = fig.add_subplot(236)
            data_cub = data

            data_cub[data >= 0] = data_cub**3
            #data_cub[data < 0] = np.sqrt(-data_cub[data < 0])

            cub.hist((data_cub - np.mean(data_cub)) / np.std(data_cub), bins=150)
            cub.set_title("cube")

            fig.tight_layout()

            st.set_y(0.95)
            fig.subplots_adjust(top = 0.85)

            fig.savefig("transformation2/{f}_{i}.png".format(f=feature_to_watch,i=i))
