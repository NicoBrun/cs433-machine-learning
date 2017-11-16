import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data

data_path = "train.csv"

y_binary,input_data,ids = load_csv_data(data_path)

indexes = [[], [], [], []]

for ind, item in enumerate(input_data):
    indexes[int(item[22])].append(ind)


for i in range(0,4) :
    y_jet = y_binary[indexes[i]]
    y_jet_0 = y_jet[y_jet == 0]
    y_jet_1 = y_jet[y_jet == 1]
    print(" jet {i} :% boson  = {b}, % something else = {s}".format(b=len(y_jet_0)/len(y_jet), s = len(y_jet_1)/len(y_jet), i= i))