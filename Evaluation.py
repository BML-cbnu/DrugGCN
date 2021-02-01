import sys, yaml
from misc import NDCG
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from Validation import createFolder

def main():
    createFolder('Result')
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)
        
    Respond_data = config["Respond_data"]
    Evaluation = config["Evaluation"]
    data1 = pd.read_csv(Respond_data)
    data1.drop(['Unnamed: 0'], axis='columns', inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(data1)

    n_fold = 3
    NDCG_List = [1,2,5,10,20,50]
    NDCG_201_fold = []

    for cv in range(n_fold):
        CV = np.load("Result/{}_CV_{}.npz".format(Evaluation,cv))
        CV_test = CV['Y_true']
        CV_pred = CV["Y_pred"]

        Y_test = CV_test
        Y_test = scaler.inverse_transform(Y_test)
        NDCG_201 = []
        for i in NDCG_List:
            ndcg_201 = NDCG(CV_test, CV_pred, i)
            NDCG_201.append(ndcg_201.sum()/ndcg_201.shape[0])

        NDCG_201_fold.append(NDCG_201)

    NDCG_Sum = np.zeros([1,len(NDCG_List)])
    NDCG_201_fold = np.array(NDCG_201_fold)
    for cv in range(n_fold):
        NDCG_Sum += NDCG_201_fold[cv,:]

    Result_NDCG = pd.DataFrame(NDCG_Sum.reshape(len(NDCG_List),1)/3, columns =["NDCG"], index=NDCG_List)

    RMSE_fold = []
    PCC_fold = []
    SCC_fold = []

    for cv in range(n_fold):
        RMSE = []
        PCC = []
        SCC = []
        CV = np.load("Result/{}_CV_{}.npz".format(Evaluation,cv))
        CV_test = CV['Y_true']
        CV_pred = CV["Y_pred"]
        CV_test = scaler.inverse_transform(CV_test)
        CV_pred = scaler.inverse_transform(CV_pred)

        for i in range(data1.shape[1]):
            y_test = CV_test[:, i]
            y_pred = CV_pred[:, i]
            y_pred = y_pred[~np.isnan(y_test)]
            y_test = y_test[~np.isnan(y_test)]

            RMSE.append(mean_squared_error(y_test,y_pred))
            PCC.append(stats.pearsonr(y_test,y_pred)[0])
            SCC.append(stats.spearmanr(y_test,y_pred)[0])

        RMSE_fold.append(RMSE)
        PCC_fold.append(PCC)
        SCC_fold.append(SCC)

    np.array(RMSE_fold).shape
    RMSE_Sum = np.zeros([1,data1.shape[1]])
    PCC_Sum = np.zeros([1,data1.shape[1]])
    SCC_Sum = np.zeros([1,data1.shape[1]])

    RMSE_fold = np.array(RMSE_fold)
    PCC_fold = np.array(PCC_fold)
    SCC_fold = np.array(SCC_fold)

    for cv in range(n_fold):
        RMSE_Sum += RMSE_fold[cv,:]
        PCC_Sum += PCC_fold[cv,:]
        SCC_Sum += SCC_fold[cv,:]

    Result_RPC = pd.DataFrame(RMSE_Sum.reshape(data1.shape[1],1)/3, columns = ["RMSE"])
    Result_RPC["PCC"] = PCC_Sum.reshape(data1.shape[1],1)/3
    Result_RPC["SCC"] = SCC_Sum.reshape(data1.shape[1],1)/3

    Result_RPC.to_csv("Result/{}_RPC.csv".format(Evaluation),mode = 'w')
    Result_NDCG.to_csv("Result/{}_NDCG.csv".format(Evaluation), mode ='w')
    
if __name__ == '__main__':
    main()




