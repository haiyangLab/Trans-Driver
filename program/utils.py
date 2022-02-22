from __future__ import division, print_function
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path, sep=',')
    list = df.columns.values.tolist()
    list.remove('Hugo_Symbol')
    list.remove('class')
    ss = StandardScaler()
    df[list] = ss.fit_transform(df[list])
    hg = df['Hugo_Symbol']
    targets = df['class']
    features = df.drop(['Hugo_Symbol', 'class'], axis=1)
    x = features.values
    y = targets.values
    y = y.astype(float)
    y = np.reshape(y, [len(y), 1])
    torch_data = MyDataset(x, y)

    return torch_data, hg


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)


