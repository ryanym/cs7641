import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import itertools

plt.style.use('seaborn-whitegrid')



def plot_tune_curve(f1_train, f1_test, xvals, algor_name=None, xlabel=None, dataset_name=None):
    plt.figure()
    plt.plot(xvals, f1_test, 'o-', color='g', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color='b', label='Train F1 Score')

    plt.ylabel('Model F1 Score')
    # plt.xlabel('No. Neighbors')
    plt.xlabel(xlabel)
    plt.title(algor_name + ' F1 Score: ' + dataset_name)

    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('figures/' + 'f1_score_' + '_'.join(dataset_name.split()) + '.png')
    plt.savefig('figures/' + '_'.join(algor_name.split()) + '_f1_score_' + '_'.join(dataset_name.split()) + '.png')



def plot_learning_curve(train_sizes, train_mean, train_std, cv_mean, cv_std, algor_name=None, dataset_name=None):
    plt.figure()
    plt.title(algor_name + " Learning Curve: " + dataset_name)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2 * train_std, train_mean + 2 * train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('figures/' + '_'.join(algor_name.split()) + '_learning_curve_' + '_'.join(dataset_name.split()) + '.png')



def plot_learning_prediction_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, algor_name=None, dataset_name=None):
    plt.figure()
    plt.title(algor_name + " Modeling Time: " + dataset_name)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2 * fit_std, fit_mean + 2 * fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.1, color="g")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_mean, 'o-', color="g", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.tight_layout()
    # plt.savefig('figures/' + 'time_' + '_'.join(dataset_name.split()) + '.png')
    plt.savefig('figures/' + '_'.join(algor_name.split()) + '_time_' + '_'.join(dataset_name.split()) + '.png')



def plot_confusion_matrix(cm, classes, normalize=False, algor_name=None, dataset_name=None, cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(algor_name + ' Confusion Matrix: ' + dataset_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # plt.savefig('figures/' + 'confusion_matrix_' + '_'.join(dataset_name.split()) + '.png')
    plt.savefig('figures/' + '_'.join(algor_name.split()) + '_confusion_matrix_' + '_'.join(dataset_name.split()) + '.png')



# def final_classifier_evaluation(clf, X_train, X_test, y_train, y_test):
#     start_time = timeit.default_timer()
#     clf.fit(X_train, y_train)
#     end_time = timeit.default_timer()
#     training_time = end_time - start_time
#
#     start_time = timeit.default_timer()
#     y_pred = clf.predict(X_test)
#     end_time = timeit.default_timer()
#     pred_time = end_time - start_time
#
#     auc = roc_auc_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#
#     print("Model Evaluation Metrics Using Untouched Test Dataset")
#     print("*****************************************************")
#     print("Model Training Time (s):   " + "{:.5f}".format(training_time))
#     print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
#     print("F1 Score:  " + "{:.2f}".format(f1))
#     print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
#     print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
#     print("*****************************************************")
#     plt.figure()
#     plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
#     plt.show()