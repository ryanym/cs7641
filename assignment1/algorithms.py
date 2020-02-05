import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC





def tune_KNN(X_train, y_train, X_test, y_test, neighbors_list):
    f1_test = []
    f1_train = []
    # klist = np.linspace(1,250,25).astype('int')
    # klist = np.linspace(1, 50, 50).astype('int')
    for i in neighbors_list:
        clf = kNN(n_neighbors=i,
                  n_jobs=-1,
                  algorithm='auto',
                  p=2,
                  random_state=0
                  )
        # print(X_train.shape, y_train.shape)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    return f1_test, f1_train


# TODO get best parameters for KNN


def tune_decision_tree(X_train, y_train, X_test, y_test, depth_list):
    f1_test = []
    f1_train = []
    # max_depth = list(range(1, 31))
    for i in depth_list:
        clf = DecisionTreeClassifier(max_depth=i, min_samples_leaf=1, criterion='entropy', random_state=0)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
    return f1_train, f1_test
    # plt.plot(max_depth, f1_test, 'o-', color='r', label='Test F1 Score')
    # plt.plot(max_depth, f1_train, 'o-', color='b', label='Train F1 Score')
    # plt.ylabel('Model F1 Score')
    # plt.xlabel('Max Tree Depth')
    #
    # plt.title(title)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

def decisition_tree_GridSearchCV(X_train, y_train):
    #parameters to search:
    #20 values of min_samples leaf from 0.5% sample to 5% of the training data
    #20 values of max_depth from 1, 20
    start_leaf_n = round(0.005 * len(X_train))
    end_leaf_n = round(0.05 * len(X_train))
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n, end_leaf_n, 20).round().astype('int'),
                  'max_depth': np.arange(1, 20)}

    dt = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=param_grid, cv=10)
    dt.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(dt.best_params_)
    # return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf']
    return dt

def tune_boosted_tree(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, title, estimator_list):
    f1_test = []
    f1_train = []
    # n_estimators = np.linspace(1, 250, 40).astype('int')
    for i in estimator_list:
        clf = GradientBoostingClassifier(n_estimators=i, max_depth=int(max_depth / 2),
                                         min_samples_leaf=int(min_samples_leaf / 2), random_state=0, )
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))

    # plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    # plt.plot(n_estimators, f1_train, 'o-', color='b', label='Train F1 Score')
    # plt.ylabel('Model F1 Score')
    # plt.xlabel('No. Estimators')
    #
    # plt.title(title)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
def boosted_tree_GridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    #parameters to search:
    #n_estimators, learning_rate, max_depth, min_samples_leaf
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n,end_leaf_n,3).round().astype('int'),
                  'max_depth': np.arange(1,4),
                  'n_estimators': np.linspace(10,100,3).round().astype('int'),
                  'learning_rate': np.linspace(.001,.1,3)}

    boostedTree = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid=param_grid, cv=10)
    boostedTree.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(boostedTree.best_params_)
    # return boost.best_params_['max_depth'], boost.best_params_['min_samples_leaf'], boost.best_params_['n_estimators'], boost.best_params_['learning_rate']
    return boostedTree


def tune_nn(X_train, y_train, X_test, y_test, hidden_layer_sizes):
    f1_test = []
    f1_train = []
    # hlist = np.linspace(1, 150, 30).astype('int')
    for i in hidden_layer_sizes:
        clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic',
                            learning_rate_init=0.05, random_state=0)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
    return f1_test, f1_train

    # plt.plot(hlist, f1_test, 'o-', color='r', label='Test F1 Score')
    # plt.plot(hlist, f1_train, 'o-', color='b', label='Train F1 Score')
    # plt.ylabel('Model F1 Score')
    # plt.xlabel('No. Hidden Units')
    #
    # plt.title(title)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

def nn_GridSearchCV(X_train, y_train):
    #parameters to search:
    #number of hidden units
    #learning_rate
    h_units = [5, 10, 20, 30, 40, 50, 75, 100]
    learning_rates = [0.01, 0.05, .1]
    param_grid = {'hidden_layer_sizes': h_units, 'learning_rate_init': learning_rates}

    nn = GridSearchCV(estimator = MLPClassifier(solver='adam',activation='logistic',random_state=0),
                       param_grid=param_grid, cv=10)
    nn.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(nn.best_params_)
    # return nn.best_params_['hidden_layer_sizes'], nn.best_params_['learning_rate_init']
    return nn.best_params_


def tune_svm(X_train, y_train, X_test, y_test, kernel_functions):
    f1_test = []
    f1_train = []
    kernel_func = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in kernel_functions:
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

    # xvals = ['linear', 'poly2', 'poly3', 'poly4', 'poly5', 'poly6', 'poly7', 'poly8', 'rbf', 'sigmoid']

    # plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    # plt.plot(xvals, f1_train, 'o-', color='b', label='Train F1 Score')
    # plt.ylabel('Model F1 Score')
    # plt.xlabel('Kernel Function')
    #
    # plt.title(title)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()

    return f1_train, f1_test

def svm_GridSearchCV(X_train, y_train):
    #parameters to search:
    #penalty parameter, C
    #
    Cs = [1e-4, 1e-3, 1e-2, 1e01, 1]
    gammas = [1,10,100]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    degrees = [2,3,4,5,6,7,8]
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels, 'degree': degrees}

    svm = GridSearchCV(estimator=SVC(random_state=0),
                       param_grid=param_grid, cv=10)
    svm.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(svm.best_params_)
    # return clf.best_params_['C'], clf.best_params_['gamma']
    return svm