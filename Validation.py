import numpy as np

def Validation(n_fold,X,Y):
    list_train_fold = []
    list_val_fold = []
    list_train = []
    Number = X.shape[0]//n_fold
    for i in range(X.shape[0]):
        list_train.append(i)
    
    for i in range(n_fold)[::-1]:
        list_val = []
        if i==n_fold-1:
            for j in range(Number*i,X.shape[0]):
                list_val.append(j)
            list_train_fold.append(np.setdiff1d(list_train,list_val))
            list_val_fold.append(list_val)
            
        
        if i != n_fold-1: 
            for j in range(Number*i,Number*(i+1)):
                list_val.append(j)
            list_train_fold.append(np.setdiff1d(list_train,list_val))
            list_val_fold.append(list_val)
        
    return list_train_fold, list_val_fold
            
