import numpy as np
import os
import pandas as pd


def read_data(file_name):
    path = os.getcwd() + file_name
    f = open(path, encoding='utf-8')
    data = pd.read_csv(f)
    name_list = data.columns.to_list()
    print(name_list)
    name_list=name_list[1:]
    data_x = data[name_list]
    x = data_x.values
    return x, name_list


def getDistanceRelation(r):
    res = []
    for i in range(len(r) - 1):
        j = i + 1
        kk = []
        for k in range(len(r[i])):
            kk.append(abs(r[i][k] - r[j][k]))
        res.append(kk)
    res = np.array(res)
    return res


def pre_glass():
    data, label = read_data('\\WineQT.csv')
    data = np.array(data)
    print(data.shape)
    data=data[:100000]
    Distance = data

    return Distance, label
