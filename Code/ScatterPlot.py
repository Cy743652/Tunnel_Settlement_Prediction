# # -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


filename_tl_ = "E:/Code/EEMD-LSTM--master/csv/prediction/data-189-1.csv"
data_tl_ = pd.read_csv(filename_tl_)

predict_svr = data_tl_['predict_svr']
predict_bp = data_tl_['predict_bp']
predict_elm = data_tl_['predict_elm']
predict_lstm = data_tl_['predict_lstm']
predict_emd = data_tl_['predict_emd']
predict_ceemdan = data_tl_['predict_ceemdan']
true = data_tl_['TRUE']
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
#
# xValue = list(range(0, 101))
# yValue = [x * np.random.rand() for x in xValue]
#
# plt.title(u'散点图示例', FontProperties=font)
#
# plt.xlabel('x-value')
# plt.ylabel('y-label')
# # plt.scatter(x, y, s, c, marker)
# # x: x轴坐标
# # y：y轴坐标
# # s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
# # c: 点的颜色
# # marker: 标记的样式 默认是 'o'
# plt.legend()
#
# plt.scatter(xValue, yValue, s=20, c="#ff1212", marker='o')
# plt.show()

def __linear_coefs(dataX, dataY):
   points = np.c_[dataX, dataY]
   #np.c_是按行连接两个矩阵，把两个矩阵左右相加，要求行数相等，类似于pandas中的merge()
   M = len(points)    #points的行数
   x_bar = np.mean(points[:, 0])   #取第一列，算其平均值
   sum_yx = 0
   sum_x2 = 0
   sum_delta = 0
   for i in range(M):
      x = points[i, 0]
      y = points[i, 1]
      sum_yx += y * (x - x_bar)      #不用管，做的是线性回归的事
      sum_x2 += x ** 2
   w = sum_yx / (sum_x2 - M * (x_bar ** 2))

   for i in range(M):
      x = points[i, 0]
      y = points[i, 1]
      sum_delta += (y - w * x)      #不用管，做的是线性回归的事
   b = sum_delta / M
   return w, b

# def __scatter_and_linear(self, real, predict, name, index):
def __scatter_and_linear(real, predict, name):
   w, b = __linear_coefs(real, predict)
   margin = abs(min(real) - max(real)) * 0.025     #为了把y=x这条线撑开来，使图好看

   plt.style.use('seaborn-darkgrid')
   plt.figure(1, figsize=(8, 4))  # wide

   plt.rc('font', family='Times New Roman')
   plt.rcParams["font.weight"] = "bold"

   x = np.linspace(min(real) - margin, max(real) + margin, 400)  #y=x由点阵组成
   y = x
   # plt.plot(x, y, color='gray', linewidth=2, linestyle=":")
   # plt.plot(x, y, color='#B57EC8', linewidth=3, linestyle="-.")
   plt.plot(x, y, color='#8888C6', linewidth=2, linestyle="-.")

   # plt.scatter(real, predict, s=18, color='white', edgecolors='red')
   plt.scatter(real, predict, s=80, color='#CA7DB4', edgecolors='#CA7DB4',alpha=0.5,marker='*')

   x = np.linspace(min(real), max(real), len(real))
   y = x * w + b
   # plt.plot(x, y, color='purple', linewidth=3, linestyle="--")
   plt.plot(x, y, color='#C65256', linewidth=2, linestyle="--")

   # plt.grid(True, linestyle=":", color="lightgray", linewidth=1, axis='both')
   plt.grid(True, linestyle="-", color="white", linewidth=1, axis='both')

   plt.xlabel("Real Data")
   plt.ylabel("Prediction")
   plt.xlim(min(real) - margin, max(real) + margin)
   plt.ylim(min(real) - margin, max(real) + margin)

   plt.subplots_adjust(left=0.09, bottom=0.13, right=0.995, top=0.89)

   plt.title(str(name) + ": Y=" + str(round(w, 2)) + "X+" + str(round(b, 2)))

def main():
   name = ['BP',
           'ELM',
           'SVR',
           'LSTM',
           'EMD-LSTM',
           'CEEMDAN-LSTM'
           ]

   # plt.subplot(2, 3, 1)
   # __scatter_and_linear(true,predict_bp,name[0])
   # plt.subplot(2, 3, 2)
   # __scatter_and_linear(true,predict_elm,name[1])
   # plt.subplot(2, 3, 3)
   # __scatter_and_linear(true,predict_svr,name[2])
   # plt.subplot(2, 3, 4)
   # __scatter_and_linear(true,predict_lstm,name[3])
   # plt.subplot(2, 3, 5)
   # __scatter_and_linear(true,predict_emd,name[4])
   # plt.subplot(2, 3, 6)
   __scatter_and_linear(true, predict_ceemdan, name[5])


   plt.show()

if __name__ == '__main__':
    main()

