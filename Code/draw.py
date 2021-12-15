import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
tempZH = [
    "180",
    "181",
    "182",
    "183",
    "184",
    "185",
    "186",
    "187",
    "188",
    "189",
    "190",
    "191",
    "192",
    "193",
    "194",
    "195",
    "200",
    "201",
    "205",
    "210",
    "215",
    "220",
    "225",
    "230",
    "235",
]
tempNB = [
    "551",
    "552",
    "553",
    "554",
    "555",
    "556",
    "557",
	"558",
	"559",
	"560",
	"561",
	"562",
	"563",
	"564",
	"565",
	"566",
	"567",
	"568",
	"569",
	"570",
	"571",
	"572",
	"573",
	"574",
	"575"
	# "576",
	# "578",
	# "579",
	# "580"
    ]

def draw():
    d = pd.read_csv("E:/Code/CEEMDAN-LSTM--master/csv/TLPrediction2/" + tempZH[0] + ".csv",
                    usecols=['TRUE', 'predict_ceemdan', 'predict_tl_ceemdan'])
    plt.rc('font', family='Times New Roman')
    plt.figure(1, figsize=(5,4), dpi = 400)
    plt.plot(d.TRUE, color='#DF8053', linestyle='-', linewidth=1.2, marker='p', markersize='2', label = 'True')
    plt.plot(d.predict_ceemdan, color='#A5678E', linestyle='--', linewidth=1, marker='p', markersize='2',label = 'CEEMDAN-LSTM')
    plt.plot(d.predict_tl_ceemdan, color='#5D7599', linestyle='--', linewidth=1, marker='p', markersize='2',label = 'Proposed')
    plt.legend(loc='best')
    # plt.savefig('E:\Code\CEEMDAN-LSTM--master\picture')
    plt.show()

def draw1():
    for i in range(len(tempZH)):
        d = pd.read_csv("E:/Code/CEEMDAN-LSTM--master/csv/TLPrediction2/" + tempZH[i] + ".csv",
                        usecols=['TRUE', 'predict_ceemdan', 'predict_tl_ceemdan'])
        plt.figure(1, figsize= (18,12), dpi = 600)
        plt.subplot(5,5,i+1)
        plt.figure(1, figsize=(3.5,2.5), dpi = 600)
        plt.plot(d.TRUE, color='#DF8053', linestyle='-', linewidth=2)
        plt.plot(d.predict_ceemdan, color='#A5678E', linestyle='--', linewidth=1)
        plt.plot(d.predict_tl_ceemdan, color='#5D7599', linestyle='--', linewidth=2)
        # plt.xlabel('time(days)', fontsize = 15)
        # plt.ylabel('height(mm)', fontsize = 15)
        plt.title(tempZH[i], fontsize = 15)
        plt.tight_layout()
        plt.tick_params(labelsize=14)
    plt.rc('font', family='Times New Roman')
    plt.savefig('E:/Code/CEEMDAN-LSTM--master/picture/ZH/zh')
    plt.show()

def draw2():
    for i in range(len(tempNB)):
        d = pd.read_csv("E:/Code/CEEMDAN-LSTM--master/csv/TLPrediction2/" + tempNB[i] + ".csv",
                        usecols=['TRUE', 'predict_ceemdan', 'predict_tl_ceemdan'])
        plt.figure(1, figsize= (18,12), dpi = 600)
        plt.subplot(5,5,i+1)
        plt.figure(1, figsize=(3.5,2.5), dpi = 600)
        plt.plot(d.TRUE, color='#DF8053', linestyle='-', linewidth=2)
        plt.plot(d.predict_ceemdan, color='#A5678E', linestyle='--', linewidth=1)
        plt.plot(d.predict_tl_ceemdan, color='#5D7599', linestyle='--', linewidth=2)
        # plt.xlabel('time(days)', fontsize=15)
        # plt.ylabel('height(mm)', fontsize=15)
        plt.title(tempNB[i], fontsize=15)
        plt.tight_layout()
        plt.tick_params(labelsize=14)
    plt.rc('font', family='Times New Roman')
    plt.savefig('E:\\Code\\CEEMDAN-LSTM--master\\picture\\NB\\nb')
    plt.show()

def draw3():
    d = pd.read_csv('E:/Code/TunnelSettlementPrediction/csv/prediction/result-zh.csv',
                    usecols=['TRUE', 'svr', 'bp', 'elm', 'dt', 'rf', 'knn', 'mlp', 'gru', 'lstm', 'emd_lstm', 'ceemdan_lstm'])

    plt.rc('font', family='Times New Roman')
    plt.figure(1, figsize=(18, 6), dpi=600)

    # plt.plot(d.TRUE, color='#000000', label='True', linewidth=2, linestyle='-')
    # plt.plot(d.svr, color='#E8EADC', label='SVR', linewidth=1, linestyle='--')
    # plt.plot(d.bp, color='#D6C782', label='BP', linewidth=1, linestyle='--')
    # plt.plot(d.elm, color='#D39394', label='ELM', linewidth=1, linestyle='--')
    # plt.plot(d.dt, color='#71838F', label='DT', linewidth=1, linestyle='--')
    # plt.plot(d.rf, color='#E5CC96', label='RF', linewidth=1, linestyle='--')
    # plt.plot(d.knn, color='#87E0FF', label='KNN', linewidth=1, linestyle='--')
    # plt.plot(d.mlp, color='#CFEADC', label='MLP', linewidth=1, linestyle='--')
    # plt.plot(d.gru, color='#FEDEE1', label='GRU', linewidth=1, linestyle='--')
    # plt.plot(d.lstm, color='#FFFF00', label='LSTM', linewidth=1, linestyle='--')
    # plt.plot(d.emd_lstm, color='#25D4D0', label='EMD-LSTM', linewidth=1, linestyle='--')
    # plt.plot(d.ceemdan_lstm, color='#F20D0D', label='CEEMDAN-LSTM', linewidth=2, linestyle='-')

    plt.plot(d.TRUE, color='#000000', linewidth=2, linestyle='-')
    plt.plot(d.svr, color='#E8EADC', linewidth=1, linestyle='--')
    plt.plot(d.bp, color='#D6C782', linewidth=1, linestyle='--')
    plt.plot(d.elm, color='#D39394', linewidth=1, linestyle='--')
    plt.plot(d.dt, color='#71838F', linewidth=1, linestyle='--')
    plt.plot(d.rf, color='#E5CC96', linewidth=1, linestyle='--')
    plt.plot(d.knn, color='#87E0FF', linewidth=1, linestyle='--')
    plt.plot(d.mlp, color='#CFEADC',  linewidth=1, linestyle='--')
    plt.plot(d.gru, color='#FEDEE1',  linewidth=1, linestyle='--')
    plt.plot(d.lstm, color='#FFFF00', linewidth=1, linestyle='--')
    plt.plot(d.emd_lstm, color='#F20D0D', linewidth=2, linestyle='-')
    plt.plot(d.ceemdan_lstm, color='#25D4D0', linewidth=1, linestyle='--')

    # plt.grid(True, linestyle=':', color='gray', linewidth='0.5', axis='both')
    plt.xlabel('time(days)', fontsize=20)
    plt.ylabel('height(mm)', fontsize=20)
    plt.title('180', fontsize=20)
    # plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.tick_params(labelsize=16)
    plt.savefig('E:\\Code\\TunnelSettlementPrediction\\picture\\result-zh')

    plt.show()

def main():
    # draw()
    # draw1()
    # draw2()
    draw3()

if __name__ == '__main__':
    main()