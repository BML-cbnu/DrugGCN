import sys
import yaml
import numpy as np
import pandas as pd
from Validation import Validation
from KRL.KRL import KRL 
from misc import Precision
from misc import NDCG
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.load(f)    
        
    n_fold = config["n_fold"]
    test_size = config["test_size"]
    Respond_data = config["Respond_data"]
    Gene_data = config["Gene_data"]
    krl_k = config["krl_k"]
    n_jobs = config["n_jobs"]
    Gamma = config["Gamma"]
    if type(Gamma) == str and Gamma == "None":
        Gamma = None
    Lambda = config["Lambda"]
    verbose = config["verbose"]
    
    X = pd.read_csv(Gene_data)
    Y = pd.read_csv(Respond_data)
    X.drop(['Unnamed: 0'], axis='columns', inplace=True)
    Y.drop(['Unnamed: 0'], axis='columns', inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(Y)
    Y = scaler.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                          test_size=test_size,
                                                          shuffle=True,
                                                          random_state=20)

    for cv in range(n_fold):
        
        list_train, list_val = Validation(n_fold, X_train, Y_train)
        
        X_train_cv = np.array(X_train)[list_train[cv]]
        X_test_cv = np.array(X_test)
        Y_train_cv = np.array(Y_train)[list_train[cv]]
        Y_test_cv = np.array(Y_test)
        Y_pred = KRL(X_train_cv, Y_train_cv, X_test_cv, krl_k,Lambda, Gamma, n_jobs, verbose)

        np.savez(('Result/KRL_CV{}'.format(cv)), Y_true=Y_test_cv, Y_pred=Y_pred)
        
        
if __name__ == '__main__':
    main()

