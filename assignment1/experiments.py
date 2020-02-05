import numpy as np
import timeit
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from algorithms import *
from utils import *



def run_cross_validation(clf, X, y, train_sizes):
    cv_scores = []
    for size in train_sizes:
        idx = np.random.randint(X.shape[0], size=size)
        X_sub = X.iloc[idx, :]
        y_sub = y.iloc[idx, :]

        score = cross_validate(clf, X_sub, y_sub, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)
        cv_scores.append(score)

    train_mean = np.zeros(len(train_sizes))
    train_std = np.zeros(len(train_sizes))
    test_mean = np.zeros(len(train_sizes))
    test_std = np.zeros(len(train_sizes))
    fit_time_mean = np.zeros(len(train_sizes))
    fit_time_std = np.zeros(len(train_sizes))
    pred_time_mean = np.zeros(len(train_sizes))
    pred_time_std = np.zeros(len(train_sizes))

    for i, score in enumerate(cv_scores):
        train_mean[i] = np.mean(score['train_score'])
        train_std[i] = np.std(score['train_score'])
        test_mean[i] = np.mean(score['test_score'])
        test_std[i] = np.std(score['test_score'])
        fit_time_mean[i] = np.mean(score['fit_time'])
        fit_time_std[i] = np.std(score['fit_time'])
        pred_time_mean[i] = np.mean(score['score_time'])
        pred_time_std[i] = np.std(score['score_time'])

    return train_mean, train_std, test_mean, test_std, fit_time_mean, fit_time_std, pred_time_mean, pred_time_std


def classifier_eval(clf, X_train, y_train, X_test, y_test):
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    start_time = timeit.default_timer()
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    print("Model Name: " + clf.__class__.__name__)
    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy))
    print("Precision: " + "{:.2f}".format(precision))
    print("AUC:       " + "{:.2f}".format(auc))
    print("Recall:    " + "{:.2f}".format(recall))

    return y_test, y_pred


def run_decision_tree_exp(X_train, y_train, X_test, y_test, dataset_name):

    tree_depth = list(range(1, 31))
    f1_train, f1_test = tune_decision_tree(X_train, y_train, X_test, y_test, tree_depth)
    plot_tune_curve(f1_train, f1_test, tree_depth, title='Decision Tree', xlabel='Max Depth', dataset_name=dataset_name)

    clf = decisition_tree_GridSearchCV(X_train, y_train)
    estimator = clf.best_estimator_

    train_sizes = (np.linspace(.05, 1.0, 20) * len(y)).astype('int')
    train_mean, train_std, test_mean, test_std, fit_time_mean, fit_time_std, pred_time_mean, pred_time_std = \
        run_cross_validation(clf, X_train, y_train, train_sizes)

    plot_learning_curve(train_sizes, train_mean, train_std, fit_time_mean, fit_time_std, dataset_name)
    # train_samp_phish, DT_train_score_phish, DT_fit_time_phish, DT_pred_time_phish = plot_learning_curve(estimator_phish,
    #                                                                                                     X1_train,
    #                                                                                                     Y1_train,
    #                                                                                                     title="Decision Tree Phishing Data")
    final_classifier_evaluation(estimator_phish, X1_train, X1_test, Y1_train, Y1_test)

def run_experiments():
    pass
