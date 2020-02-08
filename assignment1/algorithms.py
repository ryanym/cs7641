import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')




def tune_knn(X_train, y_train, X_test, y_test, neighbors_list):
    f1_test = []
    f1_train = []
    for i in neighbors_list:
        print('kNN: {0}/{1}'.format(i, max(neighbors_list)))
        clf = kNN(n_neighbors=i,
                  n_jobs=-1,
                  algorithm='auto',
                  p=2,
                  )
        # print(X_train.shape, y_train.shape)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    return f1_train, f1_test


def knn_GridSearchCV(X_train, y_train):
    neighbors_list = np.arange(1, 30)
    weights = ['uniform', 'distance']
    param_grid = {'n_neighbors': neighbors_list,
                  'weights': weights}

    knn = GridSearchCV(estimator=kNN(n_jobs=-1), param_grid=param_grid, cv=10)
    knn.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(knn.best_params_)
    return knn

def tune_decision_tree(X_train, y_train, X_test, y_test, depth_list):
    f1_test = []
    f1_train = []
    for i in depth_list:
        print('DT: {0}/{1}'.format(i, max(depth_list)))
        clf = DecisionTreeClassifier(max_depth=i, min_samples_leaf=1, criterion='entropy', random_state=0)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
    return f1_train, f1_test


def decisition_tree_GridSearchCV(X_train, y_train):

    start_leaf_n = round(0.005 * len(X_train))
    end_leaf_n = round(0.05 * len(X_train))
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n, end_leaf_n, 20).round().astype('int'),
                  'max_depth': np.arange(1, 20)}

    dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=param_grid, cv=10, n_jobs=-1)
    dt.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(dt.best_params_)
    # return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf']
    return dt

def tune_boosted_tree(X_train, y_train, X_test, y_test,  estimator_list):
    f1_test = []
    f1_train = []
    max_depth = 5
    min_samples_leaf = 30
    for i in estimator_list:
        print('BDT: {0}/{1}'.format(i, max(estimator_list)))
        clf = GradientBoostingClassifier(n_estimators=i, max_depth=int(max_depth / 2),
                                         min_samples_leaf=int(min_samples_leaf / 2), random_state=0, )
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
    return f1_train, f1_test


def boosted_tree_GridSearchCV(X_train, y_train):

    start_leaf_n = round(0.005 * len(X_train))
    end_leaf_n = round(0.05 * len(X_train))
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n,end_leaf_n,3).round().astype('int'),
                  'max_depth': np.arange(1,4),
                  'n_estimators': np.linspace(10,100,3).round().astype('int'),
                  'learning_rate': np.linspace(.001,.1,3)}

    boostedTree = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid=param_grid, cv=10, n_jobs=-1)
    boostedTree.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(boostedTree.best_params_)
    return boostedTree


def tune_nn(X_train, y_train, X_test, y_test, hidden_layer_sizes):
    f1_test = []
    f1_train = []
    for i in hidden_layer_sizes:
        print('NN: {0}/{1}'.format(i, max(hidden_layer_sizes)))
        clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic',
                            learning_rate_init=0.05, random_state=0)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
    return f1_train, f1_test


def nn_GridSearchCV(X_train, y_train):
    h_units = [5, 10, 20, 30, 40, 50, 75, 100]
    learning_rates = [0.01, 0.05, .1]
    param_grid = {'hidden_layer_sizes': h_units, 'learning_rate_init': learning_rates}

    nn = GridSearchCV(estimator = MLPClassifier(solver='adam',activation='logistic',random_state=0),
                       param_grid=param_grid, cv=10)
    nn.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(nn.best_params_)
    return nn


def tune_svm(X_train, y_train, X_test, y_test, kernel_functions):
    f1_test = []
    f1_train = []
    for i in kernel_functions:
        print('SVM: {0}'.format(i))
        if 'poly' in i:
            j = int(i.split('poly')[1])
            clf = SVC(kernel='poly', degree=j, random_state=0)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
        else:
            clf = SVC(kernel=i, random_state=0)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))

    return f1_train, f1_test


def svm_GridSearchCV(X_train, y_train):

    print('in SVM grid search cv')
    C = [1e-2, 1e-1, 1]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    degrees = [2, 3, 4]
    param_grid = {'C': C,  'kernel': kernels, 'degree': degrees}

    svm = GridSearchCV(estimator=SVC(random_state=0),
                       param_grid=param_grid, cv=10, n_jobs=-1)
    svm.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(svm.best_params_)

    return svm