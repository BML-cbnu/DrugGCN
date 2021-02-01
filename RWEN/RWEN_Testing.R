source("RWEN.R")
library(glmnet)
args = commandArgs(TRUE)
EXP <- read.csv(file=args[1],header = F)
DRUG <- read.csv(file=args[2],header = F)
EXP_Dim = dim(EXP)
DRUG_Dim = dim(DRUG)
EXP = as.matrix(EXP)
DRUG = as.matrix(DRUG)
n_fold = as.numeric(args[4])
Train_num = EXP_Dim[1]*(1-as.numeric(args[5]))

y_pred = array( dim = DRUG_Dim[2] )
y_true = array( dim = DRUG_Dim[2] )
X_train <-EXP[1:Train_num,]
X_test <- EXP[Train_num:EXP_Dim[1],]
Y_train <-DRUG[1:Train_num,]
Y_test <- DRUG[Train_num:EXP_Dim[1],]
for (cv in 1:n_fold){
  for (i in 1:DRUG_Dim[2]){
    y_train = Y_train[,i]
    x_train = X_train[!is.na(y_train),]
    y_train = y_train[!is.na(y_train)]

    list_train = Validation(3, X_train, Y_train)
  
    Result = IterWeight(X.train = x_train[list_train[[cv,1]],], X.test = X_test,
                    y.train = y_train[list_train[[cv,1]]], y.test = Y_test[,i])
    pred =Result['yhat.test']
    y_pred[i] = pred
  }
  Dic = paste0(args[3],"/RWEN_CV_",cv-1,".csv")
  write.csv(y_pred,Dic)
}



 

