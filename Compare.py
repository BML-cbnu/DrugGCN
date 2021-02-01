import yaml
import sys
import pandas as pd
import numpy as np
from Validation import Validation, createFolder
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from KRL.KRL import KRL 

def main():
    createFolder('Result')
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)
        
    n_fold = config["n_fold"]
    test_size = config["test_size"]
    Response_data = config["Response_data"]
    Gene_data = config["Gene_data"]
    krl_k = config["krl_k"]
    n_jobs = config["n_jobs"]
    Gamma = config["Gamma"]
    if type(Gamma) == str and Gamma == "None":
        Gamma = None
    Lambda = config["Lambda"]
    verbose = config["verbose"]
    
    data1 = pd.read_csv(Response_data)
    data1.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data2 = pd.read_csv(Gene_data)
    data2.drop(['Unnamed: 0'], axis='columns', inplace=True)
    data2=np.array(data2)

    scaler = MinMaxScaler()
    scaler.fit(data1)
    data1 = scaler.transform(data1)
    
    train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(data2, data1, 
                                                          test_size=test_size, shuffle=True, random_state=20)

    X_train = np.array(train_data_split[:]).astype(np.float32)
    X_test = np.array(test_data_split[:]).astype(np.float32)
    Y_train = np.array(train_labels_split).astype(np.float32)
    Y_test = np.array(test_labels_split[:]).astype(np.float32)

    # KernelRidge
    for cv in range(n_fold):
        model = KernelRidge()
        Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
        for i in range(Y_train.shape[1]):
            y_train = Y_train[:, i]
            x_train = X_train[~np.isnan(y_train)]
            y_train = y_train[~np.isnan(y_train)]

            train_fold, val_fold = Validation(n_fold, x_train,y_train)

            model.fit(x_train[train_fold[cv]], y_train[train_fold[cv]])
            Y_pred[:, i] = model.predict(X_test)
        np.savez(('Result/KRR_CV_{}'.format(cv)), Y_true=Y_test, Y_pred=Y_pred)

    # RandomForestRegressor
    for cv in range(n_fold):
        model = RandomForestRegressor()
        Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
        for i in range(Y_train.shape[1]):
            y_train = Y_train[:, i]
            x_train = X_train[~np.isnan(y_train)]
            y_train = y_train[~np.isnan(y_train)]

            train_fold, val_fold = Validation(n_fold, x_train,y_train)

            model.fit(x_train[train_fold[cv]], y_train[train_fold[cv]])
            Y_pred[:, i] = model.predict(X_test)
        np.savez(('Result/RF_CV_{}'.format(cv)), Y_true=Y_test, Y_pred=Y_pred)

    # MLPRegressor
    for cv in range(n_fold):
        model = MLPRegressor()
        Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
        for i in range(Y_train.shape[1]):
            y_train = Y_train[:, i]
            x_train = X_train[~np.isnan(y_train)]
            y_train = y_train[~np.isnan(y_train)]

            train_fold, val_fold = Validation(n_fold, x_train,y_train)

            model.fit(x_train[train_fold[cv]], y_train[train_fold[cv]])
            Y_pred[:, i] = model.predict(X_test)
        np.savez(('Result/MLP_CV_{}'.format(cv)), Y_true=Y_test, Y_pred=Y_pred)


    # BaggingRegressor
    for cv in range(n_fold):
        model = BaggingRegressor()
        Y_pred = np.zeros([X_test.shape[0], Y_train.shape[1]])
        for i in range(Y_train.shape[1]):
            y_train = Y_train[:, i]
            x_train = X_train[~np.isnan(y_train)]
            y_train = y_train[~np.isnan(y_train)]

            train_fold, val_fold = Validation(n_fold, x_train,y_train)

            model.fit(x_train[train_fold[cv]], y_train[train_fold[cv]])
            Y_pred[:, i] = model.predict(X_test)
        np.savez(('Result/BR_CV_{}'.format(cv)), Y_true=Y_test, Y_pred=Y_pred)

    # Kernel Ranked Learning
    for cv in range(n_fold):
        
        list_train, list_val = Validation(n_fold, X_train, Y_train)
        
        X_train_cv = np.array(X_train)[list_train[cv]]
        X_test_cv = np.array(X_test)
        Y_train_cv = np.array(Y_train)[list_train[cv]]
        Y_test_cv = np.array(Y_test)
        
        Y_pred = KRL(X_train_cv,Y_train_cv,X_test_cv,krl_k,Lambda,Gamma,n_jobs,verbose)
        
        np.savez(('Result/KRL_CV{}'.format(cv)), Y_true=Y_test_cv, Y_pred=Y_pred)

if __name__ == '__main__':
    main()




