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
    for i in range(0,4) :

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
            av.hist((data1, data2),bins= 150, histtype='barstacked', label = ["s","b"],  color = ['r', 'b'])

            av.legend(loc='upper right')
            av.set_title("Initial data")

            norm = fig.add_subplot(232)
            data_norm = (data - np.mean(data)) / np.std(data)
            norm.hist((data_norm[output==1],data_norm[output == 0]), bins=150,histtype='barstacked', label = ["s","b"],  color = ['r', 'b'])
            norm.legend(loc='upper right')
            norm.set_title("just normalization")


            log_=fig.add_subplot(233)
            data_ln = data
            data_ln[data > 0] = np.log(data[data > 0])
            data_ln[data < 0] = np.log(-data[data < 0])
            data_ln[data==0] = np.mean(data[data != 0])
            data_ln = (data_ln-np.mean(data_ln))/np.std(data_ln)
            log_.hist((data_ln[output == 1], data_ln[output == 0]), bins=150, label = ["s","b"],histtype='barstacked', color=['r', 'b'])

            log_.legend(loc='upper right')
            log_.set_title("log")




            par= fig.add_subplot(234)

            data_square = data
            #data_pareto[data > 0] = 1 / data_pareto[data > 0]
            data_square = data_square ** 6
            data_square[data == 0] = np.mean(data[data > 0])

            data_square_normalised  = (data_square - np.mean(data_square)) / np.std(data_square)

            par.hist((data_square_normalised[output == 1], data_square_normalised[output == 0]), bins=150, label = ["s","b"], histtype='barstacked', color=['r', 'b'])

            par.legend(loc='upper right')
            par.set_title("6")


            '''
            ex = fig.add_subplot(234)
            data_ex = data
            data_ex = np.exp(data_ex)
            data_ex = (data_ex - np.mean(data_ex))/np.std(data_ex)

            ex.hist((data_ex[output == 1],data_ex[output == 0]),bins=150, label = ["s","b"],histtype='bar', color=['r', 'b'])
            ex.legend(loc='upper right')
            ex.set_title("exp")
            '''
            dec= fig.add_subplot(235)
            data_decal = data

            data_decal[data >= 0] = np.sqrt(data_decal[data >= 0])
            data_decal[data < 0] = np.sqrt(-data_decal[data < 0])
            data_decal = (data_decal - np.mean(data_decal)) / np.std(data_decal)

            dec.hist((data_decal[output == 1],data_decal[output == 0]), bins=150, label = ["s","b"], histtype='barstacked', color=['r', 'b'])
            dec.legend(loc='upper right')
            dec.set_title("sqrt")


            cub = fig.add_subplot(236)
            data_cub = data

            data_cub[data >= 0] = data_cub**7
            data_cub = (data_cub - np.mean(data_cub)) / np.std(data_cub)

            cub.hist((data_cub[output == 1],data_cub[output == 0]), bins=150, label = ["s","b"], histtype='barstacked', color=['r', 'b'])
            cub.set_title("7")
            cub.legend(loc ='upper right')

            fig.tight_layout()

            st.set_y(0.95)
            fig.subplots_adjust(top = 0.85)

            fig.savefig("transfo_exp/stack_6_7_{f}_{i}.png".format(f=feature_to_watch,i=i))
