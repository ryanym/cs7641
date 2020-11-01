import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import itertools

plt.style.use('seaborn-whitegrid')



def plot_tune_curve(f1_train, f1_test, xvals, algor_name=None, xlabel=None, dataset_name=None):
    plt.figure()
    plt.plot(xvals, f1_test, 'o-', color='g', label='Test Accuracy')
    plt.plot(xvals, f1_train, 'o-', color='b', label='Train Accuracy')

    plt.ylabel('Model Accuracy')
    # plt.xlabel('No. Neighbors')
    plt.xlabel(xlabel)
    plt.title(algor_name + ' Accuracy: ' + dataset_name)

    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('figures/' + 'f1_score_' + '_'.join(dataset_name.split()) + '.png')
    plt.savefig('figures/' + '_'.join(algor_name.split()) + '_accuracy_' + '_'.join(dataset_name.split()) + '.png')



def plot_learning_curve(train_sizes, train_mean, train_std, cv_mean, cv_std, algor_name=None, dataset_name=None):
    plt.figure()
    plt.title(algor_name + " Learning Curve: " + dataset_name)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Accuracy")
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


# Plot utils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')

IMAGE_DIR = 'figures/'


def plot_clusters(ax, component1, component2, df, name):
    # Plot input data onto first two components at given axes
    y_colors = sns.color_palette('hls', len(np.unique(df['y'])))
    c_colors = sns.color_palette('hls', len(np.unique(df['c'])))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=y_colors, data=df, legend='full', alpha=0.3, ax=ax[0])
    sns.scatterplot(x=component1, y=component2, hue='c', palette=c_colors, data=df, legend='full', alpha=0.3, ax=ax[1])

    # Set titles
    ax[0].set_title('True Clusters represented with {}'.format(component1[:-1].upper()))
    ax[1].set_title('{} Clusters represented with {}'.format(name.upper(), component1[:-1].upper()))

    # Set axes limits
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)


def plot_components(component1, component2, df, name):
    # Create figure and plot input data onto first two components
    plt.figure()
    colors = sns.color_palette('hls', len(np.unique(df['y'])))
    sns.scatterplot(x=component1, y=component2, hue='y', palette=colors, data=df, legend='full', alpha=0.3)

    # Annotate standard deviation arrows for the two components
    plt.annotate('', xy=(np.std(df[component1]), 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    plt.annotate('', xy=(0, np.std(df[component2])), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='orange', lw=3))

    # Set title and axes limits
    plt.title('{} Transformation with first 2 components and true labels'.format(name.upper()))
    xlim = 1.1 * np.max(np.abs(df[component1]))
    ylim = 1.1 * np.max(np.abs(df[component2]))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)


def plot_multiple_random_runs(x_axis, y_axis, label):

    # Plot mean with area between mean + std and mean - std filled of the same color
    y_mean, y_std = np.mean(y_axis, axis=0), np.std(y_axis, axis=0)
    plot = plt.plot(x_axis, y_mean, '-o', markersize=1, label=label)
    plt.fill_between(x_axis, y_mean - y_std, y_mean + y_std, alpha=0.1, color=plot[0].get_color())


def plot_ic_bars(ic, ic_label, cv_types, k_range, ax):

    bar_width = 0.2  # width of a bar

    # For each covariance type, plot a histogram of different color
    for i, cv_type in enumerate(cv_types):
        x = k_range + bar_width / 2 * (i - 2)  # compute x
        ax.bar(x, ic[i * len(k_range):(i + 1) * len(k_range)], width=bar_width, label=cv_type)

    # Set x values, y limit aand legend
    ax.set_xticks(k_range)
    ax.set_ylim([ic.min() * 1.01 - .01 * ic.max(), ic.max()])
    ax.legend(loc='best')

    # Set title and labels
    set_axis_title_labels(ax=ax, title='EM - Choosing k with the {} method'.format(ic_label),
                          x_label='Number of components k', y_label='{} score'.format(ic_label))

def save_figure(title):
    """Save Figure.

        Args:
          title (string): plot title.

        Returns:
          None.
        """
    plt.savefig(IMAGE_DIR + title)
    plt.close()


def save_figure_tight(title):
    """Save Figure with tight layout.

        Args:
          title (string): plot title.

        Returns:
          None.
        """

    plt.tight_layout()
    save_figure(title)


def set_axis_title_labels(ax, title, x_label, y_label):

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def set_plot_title_labels(title, x_label, y_label):

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
