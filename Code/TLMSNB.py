import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PyEMD import EEMD, EMD,CEEMDAN
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional
from evaluate_data import *
import keras
from keras.optimizers import *

def data_split(data, train_len, lookback_window):

    X_all = []
    Y_all = []
    data = data.reshape(-1, )
    for i in range(lookback_window, len(data)):
        X_all.append(data[i - lookback_window:i])
        Y_all.append(data[i])

    X_train = X_all[:train_len]
    X_test = X_all[train_len:]

    Y_train = Y_all[:train_len]
    Y_test = Y_all[train_len:]

    return [np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)]


def data_split_LSTM(data_regular):  # data split f
	X_train = data_regular[0].reshape(data_regular[0].shape[0], data_regular[0].shape[1], 1)
	Y_train = data_regular[1].reshape(data_regular[1].shape[0], 1)
	X_test = data_regular[2].reshape(data_regular[2].shape[0], data_regular[2].shape[1], 1)
	y_test = data_regular[3].reshape(data_regular[3].shape[0], 1)
	return [X_train, Y_train, X_test, y_test]


def load_data(file):
	dataset = pd.read_csv(file, header=0, index_col=0, parse_dates=True)

	df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
	do = df['all_time_change']  # 返回all_time_change那一列，用字典的方式
	print(do)
	full_data = []
	for i in range(0, len(do)):
		full_data.append([do[i]])

	scaler_data = MinMaxScaler(feature_range=(0, 1))
	full_data = scaler_data.fit_transform(full_data)   #归一化
	print('Size of the Dataset: ', full_data.shape)

	return full_data, scaler_data

def imf_data(data, lookback_window):
	X1 = []
	for i in range(lookback_window, len(data)):
		X1.append(data[i - lookback_window:i])
	X1.append(data[len(data) - 1:len(data)])
	X_train = np.array(X1)
	return X_train

def model_LSTM(step_num):
	model = Sequential()
	model.add(LSTM(50, input_shape=(step_num, 1)))   #已经确定10步长
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	return model

def EEMD_LSTM_Model(X_train, Y_train,i):
	filepath = 'res/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
	callbacks_list = [checkpoint]
	model = Sequential()
	# model.add(Bidirectional(LSTM(50,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]))))
	model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(Dense(50, activation='tanh'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, Y_train, epochs=20, batch_size=1, validation_split=0.1, verbose=2, shuffle=True)
	return model

temp = ["E:/Code/CEEMDAN-LSTM--master/csv/NB/data-551-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-552-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-553-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-554-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-555-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-556-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-557-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-558-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-559-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-560-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-561-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-562-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-563-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-564-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-565-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-566-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-567-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-568-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-569-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-570-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-571-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-572-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-573-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-574-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-575-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-576-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-578-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-579-1.csv",
        "E:/Code/CEEMDAN-LSTM--master/csv/NB/data-580-1.csv"
        ]

def main():

    for i in range(len(temp)):
        print("************"+temp[i])
        full_data, scaler = load_data(temp[i])

        training_set_split = int(len(full_data) * 0.95)
        lookback_window = 2


        # #数组划分为不同的数据集
        data_regular = data_split(full_data, training_set_split, lookback_window)
        y_real = scaler.inverse_transform(data_regular[3].reshape(-1, 1)).reshape(-1, )


        # # TLCEEMDAN-LSTM
        ceemdan = CEEMDAN()
        tlceemdan_imfs = ceemdan.ceemdan(full_data.reshape(-1), None, 8)
        tlceemdan_imfs_prediction = []

        test = np.zeros([len(full_data) - training_set_split - lookback_window, 1])

        i = 1
        for imf in tlceemdan_imfs:
            print('-' * 45)
            print('This is  ' + str(i) + '  time(s)')
            print('*' * 45)

            data_imf = data_split_LSTM(data_split(imf_data(imf, 1), training_set_split, lookback_window))
            test += data_imf[3]

            model = keras.models.load_model("E:/Code/CEEMDAN-LSTM--master/ModelSave/NB1/CEEMDAN-LSTM-imf" + str(i) + ".h5")
            # for l in model.layers:
            # 	print(l.name)
            # 	print(l.get_config())
            # for layer in model.layers[:1]:
            # 	layer.trainable = False
            # model.compile(optimizer=Adam(lr=0.0001), loss='mse')
            model.fit(data_imf[0], data_imf[1],epochs=20, batch_size=1, validation_split=0.1, verbose=2, shuffle=True)
            model.save("E:/Code/CEEMDAN-LSTM--master/ModelSave/NB1/CEEMDAN-LSTM-imf" + str(i) + ".h5")
            # prediction_tl = model.predict(data_imf[2])
            # tlceemdan_imfs_prediction.append(prediction_tl)
            i += 1


if __name__ == '__main__':

	main()
