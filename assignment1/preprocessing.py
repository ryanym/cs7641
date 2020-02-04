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

    data2 = pd.read_csv("data/online_shoppers_intention.csv")
    data2 = data2.dropna()

    columns_to_encode = ['VisitorType', 'Weekend', 'Month', 'Revenue']
    label_encoder = preprocessing.LabelEncoder()
    data2_encoded = data2[columns_to_encode]
    data2_encoded = data2_encoded.apply(label_encoder.fit_transform)
    data2 = data2.drop(columns_to_encode, axis=1)
    data2 = pd.concat([data2, data2_encoded], axis=1)

    if hist:
        data2.drop(['Revenue'], axis=1).hist(figsize=(20,20))
        plt.tight_layout()
        plt.savefig('figures/online_shoppers_histogram.png')

    columns_to_onehot = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
    data2 = pd.get_dummies(data2, columns=columns_to_onehot)
    X2 = data2.drop(['Revenue'], axis=1)
    y2 = data2[['Revenue']]

    return X1, y1, X2, y2


if __name__ == '__main__':
    get_data()