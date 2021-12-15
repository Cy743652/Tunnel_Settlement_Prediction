import pandas as pd
import numpy as np
from pyhht.emd import EMD
from Support.support_wavelet import *
from Support.support_VMD import *
from sklearn import preprocessing


def read_csv_hk(filename, trainNum, start_Num, ahead_num):

    dataset = pd.read_csv(filename, encoding='gbk')
    dataset = dataset.iloc[-10001 + start_Num:,-2].values
    dataset_array = preprocessing.scale(dataset)

    dataset_array = dataset_array.reshape(-1, 1)

    # 输入数据
    train_num = trainNum

    dataX1, dataX2 = [], []
    dataY1, dataY2 = [], []

    for i in range(train_num - ahead_num + 1):
        # print(i)
        a = dataset_array[i:(i + ahead_num), 0]
        dataX1.append(a)
    for j in range(train_num - ahead_num, len(dataset_array) - ahead_num):
        b = dataset_array[j:(j + ahead_num), 0]
        dataX2.append(b)

    dataY1 = dataset_array[ahead_num:train_num + 1, 0]
    dataY2 = dataset_array[train_num:, 0]

    # print(np.array(dataX1).shape)
    return np.array(dataX1),np.array(dataY1),np.array(dataX2),np.array(dataY2)

def create_data(data, train_num, time_step):

    dataX1, dataX2 = [], []
    dataY1, dataY2 = [], []

    for i in range(train_num - time_step + 1):
        # print(i)
        a = data[i:(i + time_step), 0]
        dataX1.append(a)
    for j in range(train_num - time_step, len(data) - time_step):
        b = data[j:(j + time_step), 0]
        dataX2.append(b)

    dataY1 = data[time_step:train_num + 1, 0]
    dataY2 = data[train_num:, 0]

    # print(np.array(dataX1).shape)
    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)

def load_data_emd(filename, train_Num, start_Num, ahead_num):
    print('EMD_data loading.')

    dataset = pd.read_csv(filename, encoding='gbk')
    dataset = dataset.iloc[-10001 + start_Num:, -2].values
    dataset_array = preprocessing.scale(dataset)

    # 处理预测信息，划分训练集和测试集
    dataset_array = np.array(dataset_array).reshape(-1, 1)


    decomposer = EMD(dataset_array)
    imfs = decomposer.decompose()
    # plot_imfs(targetData, imfs)
    data_decomposed = imfs.tolist()

    for h1 in range(len(data_decomposed)):
        data_decomposed[h1] = np.array(data_decomposed[h1]).reshape(-1, 1)
    for h2 in range(len(data_decomposed)):
        trainX, trainY, testX, testY = create_data(data_decomposed[h2], train_Num, ahead_num)
        dataset_imf = [trainX, trainY, testX, testY]
        data_decomposed[h2] = dataset_imf

    print('load_data complete.\n')

    return data_decomposed

def load_data_wvlt(filename, train_Num, start_Num, ahead_num):
    print('WVLT_data loading.')

    dataset = pd.read_csv(filename, encoding='gbk')
    dataset = dataset.iloc[-10001 + start_Num:, -2].values
    dataset_array = preprocessing.scale(dataset)

    # 处理预测信息，划分训练集和测试集

    wavefun = 'db1'
    dataset_array = swt_decom(dataset_array, wavefun, 3)

    for h1 in range(len(dataset_array)):
        dataset_array[h1] = np.array(dataset_array[h1]).reshape(-1, 1)
    for h2 in range(len(dataset_array)):
        trainX, trainY, testX, testY = create_data(dataset_array[h2], train_Num, ahead_num)
        dataset_wvlt = [trainX, trainY, testX, testY]
        dataset_array[h2] = dataset_wvlt

    print('load_data complete.\n')

    return dataset_array

def load_data_vmd(filename, train_Num, start_Num, ahead_num):
    print('VMD_data loading.')

    dataset = pd.read_csv(filename, encoding='gbk')
    dataset = dataset.iloc[-10001 + start_Num:, -2].values
    dataset_array = preprocessing.scale(dataset)

    # 处理预测信息，划分训练集和测试集

    wvlt_lv = 3
    VMD_level = wvlt_lv + 1
    imf_list = VMD(dataset_array, VMD_level)

    coeffs = []
    for i in range(len(imf_list)):
        imf = imf_list[i]
        for j in range(len(imf)):
            part_real = imf[j].real
            imf[j] = part_real
        coeffs.append(np.array(imf))

    for h1 in range(len(coeffs)):
        coeffs[h1] = np.array(coeffs[h1]).reshape(-1, 1)
    for h2 in range(len(coeffs)):
        trainX, trainY, testX, testY = create_data(coeffs[h2], train_Num, ahead_num)
        dataset_vmd = [trainX, trainY, testX, testY]
        coeffs[h2] = dataset_vmd

    print('load_data complete.\n')

    return coeffs