import math
from sklearn.metrics import mean_squared_error #均方误差       MSE
from sklearn.metrics import mean_absolute_error #平方绝对误差  MAE
from sklearn.metrics import r2_score#R square #调用

def deal_flag(Data, min_ramp):

    flag_temp = []

    # global minLen
    for i in range(len(Data) - 1):
        # 上升为 1.
        if (Data[i+1] - Data[i]) > min_ramp:
            flag_temp.append(1)
        # 下降为 2.
        elif (Data[i + 1] - Data[i]) < min_ramp:
            flag_temp.append(2)
        # 不变为 0.
        else:
            flag_temp.append(0)

    return flag_temp

def deal_accuracy(flag1, flag2):

    rightCount = 0
    # flag1 = flag1[1:]
    for i in range(len(flag2)):
        if flag1[i] == flag2[i]:
            rightCount = rightCount+1
    accuracy = rightCount / len(flag2)

    return accuracy * 100

def RMSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    rmse = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
    return rmse

def MAPE1(true,predict):

    L1 = int(len(true))
    L2 = int(len(predict))

    if L1 == L2:

        SUM = 0.0
        for i in range(L1):
            if true[i] == 0:
                SUM = abs(predict[i]) + SUM
            else:
                SUM = abs((true[i] - predict[i]) / true[i]) + SUM
        per_SUM = SUM * 100.0
        mape = per_SUM / L1
        return mape
    else:
        print("error")

def MAE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mae=mean_absolute_error(testY[:], testPredict[:])
    return mae


def R2(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

def Evaluate(name, name_short, true, predict, accuracy):

    rmse = RMSE1(true, predict)
    mape = MAPE1(true, predict)
    mae = MAE1(true, predict)
    r2 = R2(true, predict)

    eva_output = '\n\nMAE_'+name+': {}'.format(mae)
    eva_output += '\nRMSE_'+name+': {}'.format(rmse)
    eva_output += '\nMAPE_'+name+': {}'.format(mape)
    eva_output += '\nR2_'+name+': {}'.format(r2)
    eva_output += '\nACC_'+name+': {}'.format(accuracy)
    result_all = [name_short, mae, rmse, mape, r2, accuracy]

    return eva_output, result_all

def Evaluate_DL(name, name_short, true, predict, accuracy):

    rmse = RMSE1(true, predict)
    mape = MAPE1(true, predict)
    mae = MAE1(true, predict)
    r2 = R2(true, predict)

    eva_output = '\n\nMAE_'+name+': {}'.format(mae)
    eva_output += '\nRMSE_'+name+': {}'.format(rmse)
    eva_output += '\nMAPE_'+name+': {}'.format(mape)
    eva_output += '\nR2_'+name+': {}'.format(r2)
    eva_output += '\nACC_'+name+': {}'.format(accuracy)
    result_all = [name_short, mae, rmse, mape, r2, accuracy]

    return eva_output, result_all

