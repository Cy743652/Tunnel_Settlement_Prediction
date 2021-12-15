import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from PyEMD import EEMD, EMD,CEEMDAN

from sklearn import svm      #### SVM回归####
from elm import *     ####ELM回归
from sklearn.neural_network import MLPRegressor   ###BP回归
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional,Flatten,RNN,GRU
from keras.models import Input, Model, Sequential
from sklearn import tree        #### 决策树回归 ####
from sklearn import ensemble    #### Adaboost回归####  ####3.7GBRT回归####  ####3.5随机森林回归####
from sklearn import neighbors   #### KNN回归####

from evaluate_data import *


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


def Training_Prediction_ML(model, y_real, scaler, data, name):

	print(str(name) + ' Start.')

	model.fit(data[0], data[1])
	predict = model.predict(data[2])
	predict = scaler.inverse_transform(predict.reshape(-1, 1)).reshape(-1, )

	global result
	result += '\n\nMAE_' + name + ': {}'.format(MAE1(y_real, predict))
	result += '\nRMSE_' + name + ': {}'.format(RMSE1(y_real, predict))
	result += '\nMAPE_' + name + ': {}'.format(MAPE1(y_real, predict))
	result += '\nR2_' + name + ':{}'.format(R2(y_real, predict))
	print(str(name) + ' Complete.')

	return predict


def Training_Prediction_DL(model, y_real, scaler, data, name):

	print(str(name) + ' Start.')

	model.fit(data[0], data[1], epochs=20, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
	predict = model.predict(data[2])
	predict = scaler.inverse_transform(predict).reshape(-1, )

	global result
	result += '\n\nMAE_' + name + ': {}'.format(MAE1(y_real, predict))
	result += '\nRMSE_' + name + ': {}'.format(RMSE1(y_real, predict))
	result += '\nMAPE_' + name + ': {}'.format(MAPE1(y_real, predict))
	result += '\nR2_' + name + ':{}'.format(R2(y_real, predict))
	print(str(name) + ' Complete.')

	return predict


def model_SVR():
	return svm.SVR()


def model_ELM():
	return ELMRegressor()


def model_BPNN():
	return MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)

def model_DT():
    return tree.DecisionTreeRegressor()

def model_RF():
    return ensemble.RandomForestRegressor(n_estimators=50)

def model_KNN():
    return neighbors.KNeighborsRegressor()

def model_MLP():
    return MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=2)

def model_LSTM(step_num):
	model = Sequential()
	model.add(LSTM(50, input_shape=(step_num, 1)))   #已经确定10步长
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	return model

def model_GRU(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    x = GRU(64)(i)
    x = Dense(16, activation='linear')(x)
    o = Dense(1, activation="linear")(x)
    model = Model(inputs=[i], outputs=[o])
    model.compile(optimizer='rmsprop', loss='mse', )
    return model


def imf_data(data, lookback_window):
	X1 = []
	for i in range(lookback_window, len(data)):
		X1.append(data[i - lookback_window:i])
	X1.append(data[len(data) - 1:len(data)])
	X_train = np.array(X1)
	return X_train


def LSTM_Model(X_train, Y_train):
	model = Sequential()
	model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))   #已经确定10步长
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, Y_train, epochs=20, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
	return model


def EEMD_LSTM_Model(X_train, Y_train,i):
	filepath = 'res/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
	callbacks_list = [checkpoint]
	model = Sequential()
	# model.add(Bidirectional(LSTM(50,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]))))
	model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(Dense(50, activation='tanh'))
	model.add(Dropout(0.3))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	model.fit(X_train, Y_train, epochs=30, batch_size=1, validation_split=0.1, verbose=2, shuffle=True)

	return model

def isolutionforest(DO):
	rng = np.random.RandomState(42)
	clf = IsolationForest(random_state=rng, contamination=0.025)  # contamination为异常样本比例
	clf.fit(DO)

	DO_copy = DO
	m = 0

	pre = clf.predict(DO)
	for i in range(len(pre)):
		if pre[i] == -1:
			DO_copy = np.delete(DO_copy, i - m, 0)
			print(i)
			m += 1
	return DO_copy


def main():

	full_data, scaler = load_data('E:\\Code\\TunnelSettlementPrediction\\csv\\ZH\\data-180-1.csv')

	training_set_split = int(len(full_data) * 0.6)
	lookback_window = 2

	global result
	result = '\nEvaluation.'

	# #数组划分为不同的数据集
	data_regular = data_split(full_data, training_set_split, lookback_window)
	data_regular_DL = data_split_LSTM(data_regular)
	y_real = scaler.inverse_transform(data_regular[3].reshape(-1, 1)).reshape(-1, )

	predict_svr = Training_Prediction_ML(model_SVR(), y_real, scaler, data_regular, 'SVR')
	predict_elm = Training_Prediction_ML(model_ELM(), y_real, scaler, data_regular, 'ELM')
	predict_bp = Training_Prediction_ML(model_BPNN(), y_real, scaler, data_regular, 'BPNN')
	predict_dt = Training_Prediction_ML(model_DT(), y_real, scaler, data_regular, 'DT')
	predict_rf = Training_Prediction_ML(model_RF(), y_real, scaler, data_regular, 'RF')
	predict_knn = Training_Prediction_ML(model_KNN(), y_real, scaler, data_regular, 'KNN')
	predict_mlp = Training_Prediction_ML(model_MLP(), y_real, scaler, data_regular, 'MLP')

	# predict_RNN = Training_Prediction_DL(model_RNN(lookback_window), y_real, scaler, data_regular_DL, 'RNN')
	predict_GRU = Training_Prediction_DL(model_GRU(lookback_window), y_real, scaler, data_regular_DL, 'GRU')
	predict_LSTM = Training_Prediction_DL(model_LSTM(lookback_window), y_real, scaler, data_regular_DL, 'LSTM')


#################################################################EMD_LSTM

	emd = EMD()
	emd_imfs = emd.emd(full_data.reshape(-1), None, 8)
	emd_imfs_prediction = []


	test = np.zeros([len(full_data) - training_set_split - lookback_window, 1])

	i = 1
	for emd_imf in emd_imfs:
		print('-' * 45)
		print('This is  ' + str(i) + '  time(s)')
		print('*' * 45)

		data_imf = data_split_LSTM(data_split(imf_data(emd_imf, 1), training_set_split, lookback_window))

		test += data_imf[3]

		model = EEMD_LSTM_Model(data_imf[0], data_imf[1], i)
		emd_prediction_Y = model.predict(data_imf[2])
		emd_imfs_prediction.append(emd_prediction_Y)
		i += 1

	emd_imfs_prediction = np.array(emd_imfs_prediction)
	emd_prediction = [0.0 for i in range(len(test))]
	emd_prediction = np.array(emd_prediction)
	for i in range(len(test)):
		emd_t = 0.0
		for emd_imf_prediction in emd_imfs_prediction:
			emd_t += emd_imf_prediction[i][0]
		emd_prediction[i] = emd_t

	emd_prediction = scaler.inverse_transform(emd_prediction.reshape(-1, 1)).reshape(-1, )

	result += '\n\nMAE_emd_lstm: {}'.format(MAE1(y_real, emd_prediction))
	result += '\nRMSE_emd_lstm: {}'.format(RMSE1(y_real, emd_prediction))
	result += '\nMAPE_emd_lstm: {}'.format(MAPE1(y_real, emd_prediction))
	result += '\nR2_emd_lstm: {}'.format(R2(y_real, emd_prediction))


################################################################CEEMDAN_LSTM

	ceemdan = CEEMDAN()
	ceemdan_imfs = ceemdan.ceemdan(full_data.reshape(-1), None, 8)
	ceemdan_imfs_prediction = []

	test = np.zeros([len(full_data) - training_set_split - lookback_window, 1])

	i = 1
	for imf in ceemdan_imfs:
		print('-' * 45)
		print('This is  ' + str(i) + '  time(s)')
		print('*' * 45)

		data_imf = data_split_LSTM(data_split(imf_data(imf, 1), training_set_split, lookback_window))
		test += data_imf[3]

		model = EEMD_LSTM_Model(data_imf[0], data_imf[1], i)  # [X_train, Y_train, X_test, y_test]
		# model.summary()
		prediction_Y = model.predict(data_imf[2])
		ceemdan_imfs_prediction.append(prediction_Y)
		i += 1

	ceemdan_imfs_prediction = np.array(ceemdan_imfs_prediction)

	ceemdan_prediction = [0.0 for i in range(len(test))]
	ceemdan_prediction = np.array(ceemdan_prediction)
	for i in range(len(test)):
		t = 0.0
		for imf_prediction in ceemdan_imfs_prediction:
			t += imf_prediction[i][0]
		ceemdan_prediction[i] = t

	ceemdan_prediction = scaler.inverse_transform(ceemdan_prediction.reshape(-1, 1)).reshape(-1, )

	result += '\n\nMAE_ceemdan_lstm: {}'.format(MAE1(y_real, ceemdan_prediction))
	result += '\nRMSE_ceemdan_lstm: {}'.format(RMSE1(y_real, ceemdan_prediction))
	result += '\nMAPE_ceemdan_lstm: {}'.format(MAPE1(y_real, ceemdan_prediction))
	result += '\nR2_ceemdan_lstm: {}'.format(R2(y_real, ceemdan_prediction))

	print(result)

	###################################################################存入CSV
	real = pd.DataFrame(y_real, columns=["TRUE"])
	svr = pd.DataFrame(predict_svr, columns=["svr"])
	bp = pd.DataFrame(predict_bp, columns=["bp"])
	elm = pd.DataFrame(predict_elm, columns=["elm"])
	dt = pd.DataFrame(predict_dt, columns=["dt"])
	rf = pd.DataFrame(predict_rf, columns=["rf"])
	knn = pd.DataFrame(predict_knn, columns=["knn"])
	mlp = pd.DataFrame(predict_mlp, columns=["mlp"])
	gru = pd.DataFrame(predict_GRU, columns=["gru"])
	lstm = pd.DataFrame(predict_LSTM, columns=["lstm"])
	emd_lstm = pd.DataFrame(emd_prediction, columns=["emd_lstm"])
	ceemdan_lstm = pd.DataFrame(ceemdan_prediction, columns=["ceemdan_lstm"])
	all = pd.concat([real, svr, bp, elm, dt, rf, knn, mlp, gru, lstm, emd_lstm, ceemdan_lstm], axis=1)
	all.to_csv('E:/Code/TunnelSettlementPrediction/csv/prediction/result-zh.csv', index=False)

	###===============画图===========================

	d = pd.read_csv('E:/Code/TunnelSettlementPrediction/csv/prediction/result-zh.csv',
					usecols=['TRUE', 'svr', 'bp', 'elm', 'dt', 'rf', 'knn', 'mlp', 'gru', 'lstm', 'emd_lstm',
							 'ceemdan_lstm'])

	plt.rc('font', family='Times New Roman')
	plt.figure(1, figsize=(18, 6), dpi=600)

	plt.plot(d.TRUE, color='#000000', label='True', linewidth=2, linestyle='-')
	plt.plot(d.svr, color='#E8EADC', label='SVR', linewidth=1, linestyle='--')
	plt.plot(d.bp, color='#D6C782', label='BP', linewidth=1, linestyle='--')
	plt.plot(d.elm, color='#D39394', label='ELM', linewidth=1, linestyle='--')
	plt.plot(d.dt, color='#71838F', label='DT', linewidth=1, linestyle='--')
	plt.plot(d.rf, color='#E5CC96', label='RF', linewidth=1, linestyle='--')
	plt.plot(d.knn, color='#87E0FF', label='KNN', linewidth=1, linestyle='--')
	plt.plot(d.mlp, color='#CFEADC', label='MLP', linewidth=1, linestyle='--')
	plt.plot(d.gru, color='#FEDEE1', label='GRU', linewidth=1, linestyle='--')
	plt.plot(d.lstm, color='#FFFF00', label='LSTM', linewidth=1, linestyle='--')
	plt.plot(d.emd_lstm, color='#25D4D0', label='EMD-LSTM', linewidth=1, linestyle='--')
	plt.plot(d.ceemdan_lstm, color='#F20D0D', label='CEEMDAN-LSTM', linewidth=2, linestyle='-')

	# plt.plot(d.TRUE, color='#000000', linewidth=2, linestyle='-')
	# plt.plot(d.svr, color='#E8EADC', linewidth=1, linestyle='--')
	# plt.plot(d.bp, color='#D6C782', linewidth=1, linestyle='--')
	# plt.plot(d.elm, color='#D39394', linewidth=1, linestyle='--')
	# plt.plot(d.dt, color='#71838F', linewidth=1, linestyle='--')
	# plt.plot(d.rf, color='#E5CC96', linewidth=1, linestyle='--')
	# plt.plot(d.knn, color='#87E0FF', linewidth=1, linestyle='--')
	# plt.plot(d.mlp, color='#CFEADC', linewidth=1, linestyle='--')
	# plt.plot(d.gru, color='#FEDEE1', linewidth=1, linestyle='--')
	# plt.plot(d.lstm, color='#FFFF00', linewidth=1, linestyle='--')
	# plt.plot(d.emd_lstm, color='#25D4D0', linewidth=1, linestyle='--')
	# plt.plot(d.ceemdan_lstm, color='#F20D0D', linewidth=2, linestyle='-')

	# plt.grid(True, linestyle=':', color='gray', linewidth='0.5', axis='both')
	plt.xlabel('time(days)', fontsize=20)
	plt.ylabel('height(mm)', fontsize=20)
	plt.title('186', fontsize=20)
	# plt.legend(loc='best',fontsize=18)
	plt.tight_layout()
	plt.tick_params(labelsize=16)
	plt.savefig('E:\\Code\\TunnelSettlementPrediction\\picture\\result-zh')

	plt.show()



if __name__ == '__main__':

	main()
