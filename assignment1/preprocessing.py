import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_data(hist=False):
    data1 = pd.read_csv("data/phishing_websites.csv")
    data1.drop(['id'], axis=1, inplace=True)

    if hist:
        data1.drop(['Result'], axis=1).hist(figsize=(20, 20))
        plt.tight_layout()
        plt.savefig('figures/phishing_feature_histogram.png')

    columns_to_onehot = []
    for col in data1:
        if data1[col].unique().size > 2:
            columns_to_onehot.append(col)

    data1 = pd.get_dummies(data1, columns=columns_to_onehot)
    for i in data1:
        data1.loc[data1[i] == -1, i] = 0

    X1 = data1.drop(['Result'], axis=1)
    y1 = data1[['Result']]

    madelon_train = './data/madelon_train.data'
    madelon_train_labels = './data/madelon_train.labels'

    X2 = pd.read_csv(madelon_train, delimiter=' ', header=None)
    X2 = X2.drop([500], axis=1)
    y2 = pd.read_csv(madelon_train_labels, delimiter=' ', header=None, names=['target'])

    return X1, y1, X2, y2


if __name__ == '__main__':
    get_data()

