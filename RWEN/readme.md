## Installation

in R:

    install.packages("glmnet)

## Run RWEN code.

    rscript RWEN_Testing.R (Gene Data) (Response Data) (Path to save) (n_fold) (Test size)

1. **Gene Data**: Gene expression data of samples.
2. **Rsponse Data**: Drug Response data of samples.
3. **Path to save**
4. **n_fold**(int)
5. **Test_size**(float)

### Example
```
rscript RWEN_Testing.R C:/R/R-4.0.3/L1000_GEXP.csv C:/R/R-4.0.3/Scale_IC50.csv /Users/82107/Desktop 3 0.25
```