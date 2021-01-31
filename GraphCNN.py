import yaml
import sys, os
import tensorflow.compat.v1 as tf
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from Validation import Validation
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from lib import models, graph, coarsening, utils
from scipy.sparse import coo_matrix

            
def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    
    PPI_data = config["PPI_data"]
    Response_data = config["Response_data"]
    Gene_data = config["Gene_data"]
    n_fold = config["n_fold"]
    test_size = config["test_size"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    brelu = config["brelu"]
    pool = config["pool"]
    regularization = config["regularization"]
    dropout = config["dropout"]
    learning_rate = config["learning_rate"]
    decay_rate = config["decay_rate"]
    momentum = config["momentum"]
    Name = config["Name"]
    F = config["F"]
    K = config["K"]
    p = config["p"]
    M = config["M"]
    
    
    data_PPI = pd.read_csv(PPI_data)
    data_PPI.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_IC50 = pd.read_csv(Response_data)
    data_IC50.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_Gene = pd.read_csv(Gene_data)
    data_Gene.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data_Gene=np.array(data_Gene)

    df = np.array(data_PPI)
    A = coo_matrix(df,dtype=np.float32)
    print(A.nnz)
    graphs, perm = coarsening.coarsen(A, levels=6, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    graph.plot_spectrum(L)

    n_fold = n_fold
    PCC = []
    SPC = []
    RMSE = []

    X_train, X_test, Y_train, Y_test = train_test_split(data_Gene, data_IC50, 
                                                                  test_size=test_size, shuffle=True, random_state=20)


    for cv in range(n_fold):   
        Y_pred = np.zeros([Y_test.shape[0], Y_test.shape[1]])
        Y_test = np.zeros([Y_test.shape[0], Y_test.shape[1]])
        j = 0
        for i in range(Y.test.shape[1]):
            data1 = data_IC50.iloc[:,i]
            data1 = np.array(data1)
            data_minmax = data1[~np.isnan(data1)]
            min = data_minmax.min()
            max = data_minmax.max()
            data1 = (data1 - min) / (max - min)

            train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(data_Gene, data1, 
                                                                  test_size=test_size, shuffle=True, random_state=20)
            train_data = np.array(train_data_split[~np.isnan(train_labels_split)]).astype(np.float32)


            list_train, list_val = Validation(n_fold,train_data,train_labels_split)

            train_data_V = train_data[list_train[cv]]
            val_data = train_data[list_val[cv]]
            test_data = np.array(test_data_split[:]).astype(np.float32)
            train_labels = np.array(train_labels_split[~np.isnan(train_labels_split)]).astype(np.float32)
            train_labels_V = train_labels[list_train[cv]]
            val_labels = train_labels[list_val[cv]]
            test_labels = np.array(test_labels_split[:]).astype(np.float32)
            train_data_V = coarsening.perm_data(train_data_V, perm)
            val_data = coarsening.perm_data(val_data, perm)
            test_data = coarsening.perm_data(test_data, perm)

            common = {}
            common['num_epochs']     = num_epochs
            common['batch_size']     = batch_size
            common['decay_steps']    = train_data.shape[0] / common['batch_size']
            common['eval_frequency'] = 10 * common['num_epochs']
            common['brelu']          = brelu
            common['pool']           = pool

            common['regularization'] = regularization
            common['dropout']        = dropout
            common['learning_rate']  = learning_rate
            common['decay_rate']     = decay_rate
            common['momentum']       = momentum
            common['F']              = F
            common['K']              = K
            common['p']              = p
            common['M']              = M

            if True:
                name = Name
                params = common.copy()

            model = models.cgcnn(L, **params)
            loss, t_step = model.fit(train_data_V, train_labels_V, val_data, val_labels)

            Y_pred[:, j] = model.predict(test_data)
            Y_test[:, j] = test_labels
            j = j+1

        np.savez(('Result/GraphCNN_CV_{}'.format(cv)), Y_true=Y_test, Y_pred=Y_pred)


if __name__ == '__main__':
    main()



