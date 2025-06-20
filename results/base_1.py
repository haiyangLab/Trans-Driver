# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
# sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# numpy
import numpy as np
import random
# pandas
import pandas as pd  
from sklearn.metrics import precision_recall_curve, roc_curve

# Get training set
def train_set():
    features_train = pd.read_csv(r'../data/train_12_10.csv', sep=',')
    
    x_train = features_train.drop(['class', 'Hugo_Symbol'], axis=1)
    y_train = features_train['class']

    features_name = list(features_train.columns.values)
    for i in range(len(features_name) - 1, -1, -1): 
        if features_name[i] == 'class' or features_name[i] == 'Hugo_Symbol':
            features_name.pop(i)
    print(features_name)
    print("\nTraining set sample size:", x_train.shape)
    print("Training set label size:", y_train.shape)
    return x_train, y_train, features_name

# Get test set
def test_set():
    features_test = pd.read_csv(r'../data/cgc.csv', sep=',')
    x_test = features_test.drop(['class', 'Hugo_Symbol'], axis=1)
    y_test = features_test['class']
    gene_name = features_test['Hugo_Symbol']
    print("Test set sample size:", x_test.shape)
    print("Test set label size:", y_test.shape)
    return x_test, y_test, gene_name

# Normalization
def norm(x_train, x_test):
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test

# Convert the datasets to tensors and process them for use with the PyTorch network
def data_trans(x_train, y_train, x_test, y_test):
    train_x = torch.from_numpy(x_train.astype(np.float32))
    train_y = torch.Tensor(y_train.astype(np.float32))
    test_x = torch.from_numpy(x_test.astype(np.float32))
    test_y = torch.Tensor(y_test.astype(np.float32))
    return train_x, train_y, test_x, test_y

# Plot the loss curve
def showlossgraph(losses):
    plt.plot(losses, "ro-", label="Train loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
