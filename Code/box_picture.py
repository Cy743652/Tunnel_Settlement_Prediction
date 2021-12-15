import pandas as pd
from Part.part_advanced_pics import *

filename_tl_ = "E:\\Code\\TunnelSettlementPrediction\\csv\\prediction\\result-180.csv"
data_tl_ = pd.read_csv(filename_tl_)

predict_svr = data_tl_['svr']
predict_bp = data_tl_['bp']
predict_elm = data_tl_['elm']
predict_dt = data_tl_['dt']
predict_rf = data_tl_['rf']
predict_knn = data_tl_['knn']
predict_mlp = data_tl_['mlp']
predict_gru = data_tl_['gru']
predict_lstm = data_tl_['lstm']
predict_emd_lstm = data_tl_['emd_lstm']
predict_ceemdan_lstm = data_tl_['ceemdan_lstm']
predict_tl_ceemdan_lstm = data_tl_['tl_ceemdan_lstm']
true = data_tl_['TRUE']

###########################################################################

###########################################################################

error_svr_ = abs_sub(true, predict_svr)
error_bp_ = abs_sub(true, predict_bp)
error_elm_ = abs_sub(true, predict_elm)
error_dt_ = abs_sub(true, predict_dt)
error_rf_ = abs_sub(true, predict_rf)
error_knn_ = abs_sub(true, predict_knn)
error_mlp_ = abs_sub(true, predict_mlp)
error_gru_ = abs_sub(true, predict_gru)
error_lstm_ = abs_sub(true, predict_lstm)
error_emd_lstm_ = abs_sub(true, predict_emd_lstm)
error_ceemdan_lstm_ = abs_sub(true, predict_ceemdan_lstm)
error_tl_ceemdan_lstm_ = abs_sub(true, predict_tl_ceemdan_lstm)


plt.figure(figsize=(18, 6), dpi=600)  # 设置画布的尺寸
plt.rcParams['font.size']= 15
plt.rcParams['font.family']='Times New Roman'
plt.ylabel("Absolute Error", fontsize=18)
labels = [
          'SVR',
          'BP',
          'ELM',
          'DT',
          'RF',
          'KNN',
          'MLP',
          'GRU',
          'LSTM',
          'EMD-LSTM',
          'CEEMDAN-LSTM',
          'Proposed']
i = 0
j = -1
box = plt.boxplot(
    [
     error_svr_[i:j],
     error_bp_[i:j],
     error_elm_[i:j],
     error_dt_[i:j],
     error_rf_[i:j],
     error_knn_[i:j],
     error_mlp_[i:j],
     error_gru_[i:j],
     error_lstm_[i:j],
     error_emd_lstm_[i:j],
     error_ceemdan_lstm_[i:j],
     error_tl_ceemdan_lstm_[i:j]],
    labels=labels,
    vert=True,
    notch=True,
    showfliers=False,
    patch_artist=True,
    boxprops={'color': 'black', 'facecolor': 'pink'})

colors = [
          'aqua',
          'dodgerblue',
          'royalblue',
          'gold',
          'orange',
          'darkorange',
          'lightgreen',
          'springgreen',
           'limegreen',
          'lightcoral',
          'tomato',
          'red']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.99)
plt.grid(True, linestyle=":", color="gray", linewidth=0.5, axis='y')
plt.savefig('E:\\Code\\TunnelSettlementPrediction\\picture\\ZH\\box.png')
plt.show()

###########################################################################

