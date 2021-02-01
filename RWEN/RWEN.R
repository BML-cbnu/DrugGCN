######################################################
#
#              Iteratively Weighted Penalized Regression
#
######################################################

IterWeight <- function(y.train, X.train, y.test, X.test, 
                       alpha = 0.2, tol = 0.001, maxCount = 20,
                       left.cut.train = quantile(y.train, 1/4),
                       right.cut.train = quantile(y.train, 3/4),
                       left.cut.test = quantile(y.train, 1/4),
                       right.cut.test = quantile(y.train, 3/4),
                       nfolds4lambda = 3,
                       tailToWeight = c("left","right", "both"),
                       print.out = TRUE) {
  
  #############################################
  trnsize <- length(y.train)
  tstsize <- length(y.test)
  
  
  #================================
  #       Initial Fitting
  #================================
  
  cat("*Starting initial fit.\n")
  
  cv.linmod <- cv.glmnet(x = X.train, y= y.train, family= "gaussian", 
                         type.measure = "mse", 
                         alpha = alpha, 
                         nfolds = nfolds4lambda,
                         intercept = TRUE,
                         standardize = TRUE)
  
  lambda <- cv.linmod$lambda.min
  EN.linmod <- glmnet(x = X.train, y= y.train, family= "gaussian", 
                      lambda = lambda, alpha = alpha,
                      intercept = TRUE,
                      standardize = TRUE)
  
  cat("*Finished initial fit.\n")
  
  yhat.train <- predict(EN.linmod, newx = X.train)
  
  
  init.tail.err <- rmse(y.train, yhat.train, direction = tailToWeight, 
                        left.cut = left.cut.train, right.cut = right.cut.train)
  
  tail.err <- tol+1
  count <- 0
  
  #================================
  #       Iteratted Weighting
  #================================
  cat("*Starting iterations.\n")
  t <- proc.time()
  while((tail.err >= tol) & (count <= maxCount)) {
    
    wt <- Weights(y = y.train, yhat = yhat.train, tail = tailToWeight, 
                  left.cut = left.cut.train, right.cut = right.cut.train)
    
    cv.EN.linmod <- cv.glmnet(x = X.train, y= y.train, family= "gaussian", 
                              weights = wt,
                              type.measure = "mse", 
                              nfolds = nfolds4lambda,
                              intercept = TRUE,
                              standardize = TRUE
    )
    
    lambda <- cv.EN.linmod$lambda.min
    
    EN.linmod <- glmnet(x = X.train, y= y.train, family= "gaussian", 
                        lambda= lambda, alpha = alpha, weights = wt,
                        intercept = TRUE,
                        standardize = TRUE)
    
    yhat.train <- predict(EN.linmod, newx = X.train)
    
    tail.err <- rmse(y = y.train, yhat = yhat.train, direction = tailToWeight, 
                     left.cut = left.cut.train, right.cut = right.cut.train)
    
    count <- count + 1
    
    if(print.out){
      cat("count = ", count, ", Training Tail Error = ", tail.err,"\n")
    }
  }
  timeTaken <- proc.time() - t
  
  cat("*Finished iterations.\n")
  count <- count-1
  
  optBeta <- EN.linmod$beta
  sparsity <- length(which(optBeta!=0))
  
  #=======================================
  #              Predicted Values
  #=======================================
  
  yhat.test <- predict(EN.linmod, newx = X.test)
  
  # RMSE results
  
  rmse.all <- sqrt(sum((yhat.test-y.test)^2))/sqrt(length(y.test))
  
  rmse.left <- rmse(y = y.test, yhat = yhat.test, 
                    direction = "left",
                    left.cut = left.cut.test,
                    right.cut = right.cut.test)
  
  rmse.right <- rmse(y = y.test, yhat = yhat.test, 
                     direction = "right", 
                     left.cut = left.cut.test,
                     right.cut = right.cut.test)
  
  rmse.both <- rmse(y = y.test, yhat = yhat.test, 
                    direction = "both", 
                    left.cut = left.cut.test,
                    right.cut = right.cut.test)
  
  if(print.out){
    cat("\n====================================================\n",
        "                  Iterated Weighting                  \n",
        "\n----------------------------------------------------",
        "\nRMSE - Total                :", round(rmse.all, 4), 
        "\nRMSE - Left Tail            :", round(rmse.left, 4),
        "\nRMSE - Right Tail           :", round(rmse.right, 4),
        "\nRMSE - Both Tails           :", round(rmse.both, 4),
        "\n----------------------------------------------------",
        "\nSparsity                    :", sparsity, 
        "\nalpha                       :", round(alpha, 4), 
        "\nlambda                      :", round(lambda, 4),
        "\nNo. of Iterations           :", count,
        "\nTime Taken                  :", timeTaken["elapsed"], " seconds",
        "\n====================================================\n")
  }
  
  
  res <- list(yhat.test = yhat.test,
              finalENModel = EN.linmod,
              rmse.all = rmse.all,
              rmse.left = rmse.left,
              rmse.right = rmse.right,
              rmse.both = rmse.both,
              sparsity = sparsity,
              timeTaken = timeTaken["elapsed"]
  )
  return(res)
}



######################################################
#
#            Iterated Weighting Scheme
#
######################################################


Weights <- function(y, yhat, tail = c("left", "right", "both"),
                    left.cut = quantile(y, 1/4),
                    right.cut = quantile(y, 3/4)) {
  n <- length(y)
  diff <- abs(y-yhat)
  
  if(tail == "left") {
    w <- exp(1 + abs(diff))
    ind <- which(y > left.cut)
    #cat(w[-ind],"\n")
    w[ind] <- 0.0
    w <- (w/sum(w))*n
  }
  
  if(tail == "right") {
    w <- exp(1 + abs(diff))
    ind <- which(y < right.cut)
    w[ind] <- 0.0
    w <- (w/sum(w))*n
  }
  
  if(tail == "both") {
    w <- exp(1 + abs(diff))
    ind <- which((y > left.cut) & (y < right.cut))
    w[ind] <- 0.0
    w <- (w/sum(w))*n
  }
  return(w)
}


######################################################
#
#                RMSE Measure
#
######################################################

rmse <- function(y, yhat, direction = c("left", "right", "both"), 
                 left.cut = quantile(y, 1/4),
                 right.cut = quantile(y, 3/4)) {
  if(direction == "left") {
    lefttailed.ind <- which((y <= left.cut))
    lefttailed.n <- length(lefttailed.ind )
    SS <- sum((y[lefttailed.ind] - yhat[lefttailed.ind])^2)
    rmse <- sqrt(SS/lefttailed.n)
  }
  
  if(direction == "right") {
    righttailed.ind <- which((y >= right.cut))
    righttailed.n <- length(righttailed.ind )
    SS <- sum((y[righttailed.ind] - yhat[righttailed.ind])^2)
    rmse <- sqrt(SS/righttailed.n)
  }
  
  if(direction == "both") {
    twotailed.ind <- which((y <= left.cut) | (y >= right.cut))
    twotailed.n <- length(twotailed.ind )
    SS <- sum((y[twotailed.ind] - yhat[twotailed.ind])^2)
    rmse <- sqrt(SS/twotailed.n)
  }
  
  return(rmse)
}

Validation <- function(n_fold, X_train, Y_train){
  list_train_fold = matrix(list(),nrow =n_fold, ncol = 1)
  list_val_fold = matrix(list(),nrow = n_fold, ncol =1)
  list_train = c()
  Number = dim(X_train)[1]%/%n_fold
  for (i in 1:dim(X_train)[1])
    list_train <- c(list_train,i)
  
  for (i in 1:n_fold){
    list_val = c()
    if (i == n_fold)
    {
      for (j in (Number*(i-1))+1:(dim(X_train)[1]-Number*(i-1)))
        list_val <- c(list_val, j)
      
      list_train_fold[[i,1]] = setdiff(list_train,list_val)
      list_val_fold[[i,1]] = c(list_val)
    } 
    
    if (i !=n_fold)
    {
      for (j in (Number*(i-1))+1:Number)
        list_val = c(list_val,j)
      
      
      list_train_fold[[i, 1]] = setdiff(list_train,list_val)
      list_val_fold[[i, 1]] = list_val
    }
    
  }
  
  return (list_val_fold)
}

