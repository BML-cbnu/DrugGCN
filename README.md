## Installation

1. Clone repository.
```
git clone https://github.com/BML-cbnu/DrugGCN
cd DrugGCN
```
2. Install the requirments.
```
pip install -r requirements.txt
```
## GCN configuration

Input:

    n of samples
    p of features
    d of drugs

    Gene_data: (n * p) Gene Expression matrix.
    PPI_data: (p * p) PPI Network matrix.
    Respond_data: (n * d) Drug-Respond matrix.

    F: Number of features.
    K: List of polynomial orders. (Filter sizes)
    p: Pooling size.

Output:

    Y_pred : y * d ( y is a validation part of n )

## Reproducing our experiments
   
Edit the Configuration file to produce experiments differently.

    vim config.yaml

Run experiments.

    python GraphCNN.py config.yaml
    python Compare.py config.yaml